"""ExaoneMoE MTP head: attention kind comes from mtp_layer_types, with RoPE."""

import unittest
from types import SimpleNamespace

from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestExaoneMoEMTPAttention(CustomTestCase):
    """The head used to inherit layer 0's sliding window from layer_types and
    the NoPE rule of full layers; it is a full-attention layer with RoPE."""

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=1),
            role="test",
            ranks=SpawnRanks(world_rank=0),
        )
        self.config = SimpleNamespace(
            rms_norm_eps=1e-6,
            sliding_window=128,
            layer_types=["sliding_attention", "full_attention"],
            mtp_layer_types=["full_attention"],
        )

    def _attention(self, is_mtp):
        from sglang.srt.models.exaone_moe import ExaoneMoEAttention

        return ExaoneMoEAttention(
            config=self.config,
            hidden_size=8,
            num_heads=4,
            num_kv_heads=1,
            layer_id=0,
            max_position_embeddings=16,
            is_mtp=is_mtp,
        )

    def test_main_layer_zero_slides_with_rope(self):
        attention = self._attention(is_mtp=False)
        self.assertEqual(attention.attn.sliding_window_size, 127)
        self.assertTrue(attention.use_rope)

    def test_mtp_head_is_full_attention_with_rope(self):
        attention = self._attention(is_mtp=True)
        self.assertEqual(attention.attn.sliding_window_size, -1)
        self.assertTrue(attention.use_rope)


if __name__ == "__main__":
    unittest.main()
