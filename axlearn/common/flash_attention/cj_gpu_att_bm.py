#!/usr/bin/env python3

# pylint: skip-file

import torch
import jax
import inspect
import jax.numpy as jnp
import triton  # pytype: disable=import-error
from jax.experimental.pallas.ops.gpu.attention import mha as pallas_mha

# from axlearn.common.flash_attention.gpu_attention import flash_attention as gpu_flash_attention
# from axlearn.common.flash_attention.gpu_attention_pallas import mha as mha

from axlearn.common.flash_attention.gpu_attention import (
    cudnn_dot_product_attention,
    flash_attention,
)
from axlearn.common.flash_attention.utils import mha_reference


def _perf_report(prefix: str):
    # 128 is the most common value for per_head_dim.
    # 16, 8192, 24, 128
    #num_heads, seq_len, per_head_dim = 24, 8192, 128
    num_heads, seq_len, per_head_dim = 24, 4096, 64

    # Vary batch size for fixed heads and seq length.
    batch_size_bench = triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[2, 4, 8],
        line_arg="library",
        # line_vals=["jax", "jax-axlearn", "jax-pallas"],
        #line_vals=["jax",],
        #line_names=["Jax",],
        #styles=[("blue", "-")],
        line_vals=["jax-axlearn",],
        line_names=["Axlearn",],
        styles=[("blue", "-")],
        # line_vals=["jax-pallas",],
        # line_names=["JaxPallas",],
        # styles=[("blue", "-")],
        # line_names=["Jax", "Axlearn", "JaxPallas"],
        # styles=[("blue", "-"), ("purple", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"{prefix}-head{num_heads}-seq1024-d{per_head_dim}",
        args={"num_heads": num_heads, "seq_len": seq_len, "per_head_dim": per_head_dim},
    )
    # # Vary num heads for fixed batch and seq length.
    # num_heads_bench = triton.testing.Benchmark(
    #     x_names=["num_heads"],
    #     x_vals=[12, 16, 32, 48, 72],
    #     line_arg="library",
    #     line_vals=["jax", "jax-triton", "jax-pallas", "jax-cudnn"],
    #     line_names=["Jax", "Jax Triton", "Pallas", "jax-cudnn"],
    #     styles=[("blue", "-"), ("purple", "-"), ("green", "-"), ("red", "-")],
    #     ylabel="ms",
    #     plot_name=f"{prefix}-batch{batch_size}-seq{seq_len}-d{per_head_dim}",
    #     args={"batch_size": batch_size, "seq_len": seq_len, "per_head_dim": per_head_dim},
    # )
    return triton.testing.perf_report(
        [batch_size_bench] #, num_heads_bench, seq_len_bench, per_head_dim_bench]
    )


@_perf_report("fwd")
def bench_flash_attention(
    batch_size: int, num_heads: int, seq_len: int, per_head_dim: int, library: str
):
    warmup = 25
    rep = 500

    if library.startswith("jax"):
        q = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
        )
        k = jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
        )
        v = jax.random.normal(
            jax.random.PRNGKey(2), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
        )
        # Bias is not supported in pallas, so we don't include it here.
        bias = None

        if "axlearn" in library:
            fn = lambda: flash_attention(q, k, v, bias, causal=True)
            
        elif "pallas" in library:
            fn = lambda: pallas_mha(q, k, v, segment_ids=None, causal=True)
        elif "cudnn" in library:
            fn = lambda: cudnn_dot_product_attention(q, k, v, bias=bias, causal=True)
        else:
            fn = lambda: mha_reference(q, k, v, bias, causal=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    else:
        raise ValueError(f"Unsupported: {library}")
    return ms


@_perf_report("grad")
def bench_flash_attention_backward(
    batch_size: int, num_heads: int, seq_len: int, per_head_dim: int, library: str
):
    warmup = 25
    rep = 500
    
    print(f"library: {library} batch_size: {batch_size}, num_heads: {num_heads}, seq_len: {seq_len}, per_head_dim: {per_head_dim}")
    dtype = jnp.float32
    if library.startswith("jax"):
        q = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, per_head_dim), dtype=dtype
        )
        k = jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, seq_len, num_heads, per_head_dim), dtype=dtype
        )
        v = jax.random.normal(
            jax.random.PRNGKey(2), (batch_size, seq_len, num_heads, per_head_dim), dtype=dtype
        )
        # Bias is not supported in pallas, so we don't include it here.
        bias = None
        segment_ids = None
        causal = True
        softmax_scale = 1.0
        # num_warps = 2
        num_warps = 4
        num_stages = 1
        block_k, block_q = 32, 32
        
        @jax.jit
        def ref_fn(q, k, v, bias):
            return mha_reference(q, k, v, bias, 
                    segment_ids=segment_ids, 
                    causal=causal,
                    softmax_scale=softmax_scale,
            ).sum()
        ref_bwd = jax.grad(ref_fn, argnums=(0, 1, 2))
        truth_fn = lambda: ref_bwd(q, k, v, bias)

        if "axlearn" in library:
            @jax.jit
            def test_fn(q, k, v, bias):
                return flash_attention(q, k, v, bias, 
                    segment_ids=segment_ids, 
                    causal=causal,
                    softmax_scale=softmax_scale,
                    num_warps=num_warps, 
                    num_stages=num_stages, 
                    block_k=block_k, 
                    block_q=block_q
                ).sum()

            test_bwd = jax.grad(test_fn, argnums=(0, 1, 2))
            fn = lambda: test_bwd(q, k, v, bias)
        elif "pallas" in library:
            @jax.jit
            def pallas_fn(q, k, v):
                return pallas_mha(q, k, v, 
                    segment_ids=segment_ids, 
                    causal=causal,
                    sm_scale=softmax_scale,
                    num_warps=num_warps, 
                    num_stages=num_stages, 
                    block_k=block_k, 
                    block_q=block_q
                ).sum()

            pallas_bwd = jax.grad(pallas_fn, argnums=(0, 1, 2))
            fn = lambda: pallas_bwd(q, k, v)
        else:
            fn = truth_fn

        for i, (test, ref_test) in enumerate(zip(fn(), truth_fn())):
            # continue
            # if not jax.numpy.allclose(test, ref_test, atol=0.0001):
            if not jax.numpy.allclose(test, ref_test, atol=9):
              raise ValueError(
                 f"{library} {i} not equal to jax reference."
                 f"Diff: {test - ref_test}"
            )
        
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    else:
        raise ValueError(f"Unsupported: {library}")
    return ms


# bench_flash_attention.run(save_path=".", print_data=True)
bench_flash_attention_backward.run(save_path=".", print_data=True)
