"""Debug compiled models with TVM instrument"""

# pylint: disable=too-many-arguments
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tvm
from tvm import relax
from tvm.contrib import tvmjs
from tvm.runtime import Device, Module, Object, ShapeTuple
from tvm.runtime.relax_vm import VirtualMachine

from mlc_llm.conversation_template import ConvTemplateRegistry
from mlc_llm.interface.help import HELP
from mlc_llm.protocol.mlc_chat_config import MLCChatConfig
from mlc_llm.serve import engine_utils
from mlc_llm.support.argparse import ArgumentParser
from mlc_llm.support.auto_device import detect_device
from mlc_llm.support.style import green, red
from mlc_llm.tokenizers import Tokenizer


def _extract_metadata(mod: Module):
    return json.loads(VirtualMachine(mod, tvm.runtime.device("cpu"))["_metadata"]())


def _load_params(
    model_weight_path: str, device: Device, model_metadata: Dict[str, Any]
) -> List[tvm.nd.NDArray]:
    params, meta = tvmjs.load_ndarray_cache(model_weight_path, device)
    param_names = [param["name"] for param in model_metadata["params"]]
    assert len(param_names) == meta["ParamSize"]

    plist = []
    for param_name in param_names:
        plist.append(params[param_name])
    return plist


def _get_tvm_module(
    model_weight_path: str, lib_path: str, device: Device, instrument: tvm.runtime.PackedFunc
):
    ex = tvm.runtime.load_module(lib_path)
    vm = relax.VirtualMachine(ex, device)
    vm.set_instrument(instrument)
    metadata = _extract_metadata(ex)
    params = _load_params(model_weight_path, device, metadata)
    return vm.module, params, metadata


class DefaultDebugInstrument:
    def __init__(self):
        pass

    def reset(self):
        pass

    def __call__(self, func, name, before_run, ret_val, *args):
        pass


class DebugChat:  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: str,
        model_lib: str,
        device: Optional[str] = "auto",
        debug_instrument: Optional[Any] = None,
    ):
        self.device = detect_device(device)
        self.instrument = (
            debug_instrument if debug_instrument else DefaultDebugInstrument()
        )
        self.mod, self.params, self.metadata = _get_tvm_module(
            model, model_lib, self.device, self.instrument
        )
        self.model_path = Path(model)
        self.config_file_path = self.model_path / "mlc-chat-config.json"
        with open(self.config_file_path, mode="rt", encoding="utf-8") as file:
            self.chat_config = MLCChatConfig.model_validate_json(file.read())

        conv_template = self.chat_config.conv_template

        self.conversation = (
            ConvTemplateRegistry.get_conv_template(conv_template)
            if isinstance(conv_template, str)
            else conv_template
        )
        self.tokenizer = Tokenizer(str(self.model_path))

        self.add_sequence_func = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
        self.begin_forward_func = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
        self.end_forward_func = tvm.get_global_func("vm.builtin.kv_state_end_forward")
        self.nd_view_func = tvm.get_global_func("vm.builtin.reshape")
        self.sample_topp_from_prob_func = tvm.get_global_func("vm.builtin.sample_top_p_from_prob")

        try:
            self.embed_func = self.mod["embed"]
        except AttributeError as exc:
            raise RuntimeError("DebugChat only supports separate embedding layer") from exc

        self.prefill_func = self.mod["prefill"]
        self.decode_func = self.mod["decode"]
        self.create_kv_cache_func = None
        if self.mod.implements_function("create_flashinfer_paged_kv_cache"):
            self.create_kv_cache_func = self.mod["create_flashinfer_paged_kv_cache"]
        elif self.mod.implements_function("create_tir_paged_kv_cache"):
            self.create_kv_cache_func = self.mod["create_tir_paged_kv_cache"]
        else:
            # TODO: Support RNN KVState # pylint: disable=fixme
            # raise RuntimeError("DebugChat cannot find create KV cache function")
            self.create_kv_cache_func = self.mod["create_rnn_state"]

        self.appeared_token_freq: Dict[int, int] = {}

    def _tokenize(self, prompt: str) -> tvm.nd.array:
        tokens = engine_utils.process_prompts(prompt, self.tokenizer.encode)  # type: ignore

        # TODO: Handle ImageData in DebugChat # pylint: disable=fixme
        assert len(tokens) == 1, "DebugChat will only handle TextData for now"

        tokens = tvm.nd.array(np.array(tokens[0]).astype("int32"), device=self.device)
        return tokens

    def _embed(self, tokens: tvm.nd.array) -> Tuple[tvm.nd.NDArray, int]:
        input_len = tokens.shape[0]
        embedding = self.embed_func(tokens, self.params)
        embedding = self.nd_view_func(embedding, ShapeTuple([1, input_len, embedding.shape[1]]))
        return embedding, input_len

    def _prefill(self, embedding: tvm.nd.NDArray, input_len: int):
        seq_len_shape = ShapeTuple([input_len])
        max_num_sequence = 1

        kv_caches = self.create_kv_cache_func(
            ShapeTuple([max_num_sequence]),
            ShapeTuple([20]),
        )

        self.add_sequence_func(kv_caches, 0)
        self.begin_forward_func(kv_caches, ShapeTuple([0]), seq_len_shape)
        logits, kv_caches = self.prefill_func(embedding, kv_caches, self.params)
        self.end_forward_func(kv_caches)
        return logits, kv_caches

    def _decode(self, token: int, kv_caches: Object):
        embedding, _ = self._embed(
            tvm.nd.array(np.array([token]).astype("int32"), device=self.device)
        )
        self.begin_forward_func(kv_caches, ShapeTuple([0]), ShapeTuple([1]))
        logits, kv_caches = self.decode_func(embedding, kv_caches, self.params)
        self.end_forward_func(kv_caches)
        return logits

    def _softmax_with_temperature(self, logits: np.ndarray, temperature: float):
        # Adjust logits based on the temperature
        logits = np.array(logits) / temperature
        logits -= np.max(logits, axis=-1, keepdims=True)

        exp_logits = np.exp(logits, logits)
        exp_logits /= np.sum(exp_logits, axis=-1, keepdims=True)
        return exp_logits

    def _apply_presence_and_freq_penalty(
        self, logits: np.ndarray, presence_penalty: float, freq_penalty: float
    ):
        for token_id, freq in self.appeared_token_freq.items():
            logits[:, :, token_id] -= freq * freq_penalty + presence_penalty

    def _sample_token_from_logits(
        self,
        logits: tvm.nd.NDArray,
        *,
        temperature=1.0,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ):
        logits_np = logits.numpy()

        if presence_penalty != 0.0 or frequency_penalty != 0.0:
            self._apply_presence_and_freq_penalty(logits_np, presence_penalty, frequency_penalty)

        logits_np = self._softmax_with_temperature(logits_np, temperature)

        logits = logits.copyfrom(logits_np)
        # next_token = self.sample_topp_from_prob_func(logits, top_p, random.random())
        next_token = int(np.argmax(logits_np))
        return next_token

    def greedy_generate(
        self,
        prompt: str,
        generate_length: int,
        stop_token_ids,
    ):
        out_tokens = []
        logits_ret = []

        input_tokens = self._tokenize(prompt)
        embedding, input_len = self._embed(input_tokens)
        logits, kv_caches = self._prefill(embedding, input_len)
        next_token = self._sample_token_from_logits(logits)
        out_tokens.append(next_token)
        logits_ret.append(logits.numpy())

        for _ in range(generate_length - 1):
            logits = self._decode(next_token, kv_caches)
            next_token = self._sample_token_from_logits(logits)
            out_tokens.append(next_token)
            logits_ret.append(logits.numpy())

            if next_token in stop_token_ids:
                break

        out_text = self.tokenizer.decode(out_tokens)
        return logits_ret, out_text, out_tokens

    def loglikelihood_request(
        self,
        req,
        result,
    ):
        logits_ret = []

        input_tokens = self._tokenize(req)
        embedding, input_len = self._embed(input_tokens)
        logits, kv_caches = self._prefill(embedding, input_len)
        logits_ret.append(logits.numpy())
        
        result_tokens = self._tokenize(result)
        for id in result_tokens.numpy():
            logits = self._decode(id, kv_caches)
            logits_ret.append(logits.numpy())
        
        return logits_ret
    
    def full_output_ids(
        self,
        prompt_ids
    ):
        embedding, input_len = self._embed(tvm.nd.array(np.array(prompt_ids, dtype=np.int32).astype("int32"), device=self.device))
        seq_len_shape = ShapeTuple([input_len])
        max_num_sequence = 1

        kv_caches = self.create_kv_cache_func(
            ShapeTuple([max_num_sequence]),
            ShapeTuple([20]),
        )

        self.add_sequence_func(kv_caches, 0)
        self.begin_forward_func(kv_caches, ShapeTuple([0]), seq_len_shape)
        logits, kv_caches = self.prefill_func(embedding, kv_caches, self.params)
        self.end_forward_func(kv_caches)
        return logits.numpy()