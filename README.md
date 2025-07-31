# Squash Test: Context Window Extension Experiment Series

This repository contains a series of experiments to determine if we can **squash** (compress) components of the context window without losing intelligence.

## Experiment Series Overview

**Goal:** Determine if semantic compression can enable massive context scaling without requiring full model retraining.

**Core Question:** Can we fit 10x more information into the same context window by learning to compress the non-essential parts?

## Test 1: Context Window Extension Boundary Testing

**Hypothesis:** Existing pretrained models can handle context windows larger than their training size with graceful degradation.

**Method:** Test GPT-2 (trained on 1024 tokens) with progressively larger context sizes.

**Result:** GPT-2 fails catastrophically at 1152 tokens with "index out of range" error.

**Interpretation:** This tells us about GPT-2's specific architectural boundary conditions, NOT about the fundamental possibility of context extension. GPT-2 simply has hard-coded limits that prevent this approach.

## Next Tests in the Series

### Test 2: Custom Model with Flexible Context (Planned)
- Build small transformer from scratch in pure Python (no PyTorch)
- Train on variable context lengths from the start
- Test if models designed for flexibility can handle context extension
- **Goal:** Prove that context extension is architecturally possible

### Test 3: Compression Learning (Planned)
- Train encoder-decoder compression on our flexible model
- Test semantic compression ratios while maintaining task performance
- **Goal:** Validate the "squashing" approach with learned compression

### Test 4: End-to-End Context Multiplication (Planned)
- Combine flexible architecture + compression training
- Achieve 10:1 context multiplication (200K effective from 20K actual)
- **Goal:** Demonstrate practical context scaling solution

## Why This Matters

Current AI models hit context window limits that prevent:
- Genomic-scale analysis (3 billion base pairs)
- Complete scientific literature analysis
- Full conversation history retention
- Complex multi-document reasoning

**If we can prove context "squashing" works, we solve these problems without waiting for hardware breakthroughs.**

## Technical Approach

**The Squashing Strategy:**
1. **Compress** non-essential context into dense representations
2. **Preserve** essential semantic relationships for task performance
3. **Extend** effective context window by 10x or more
4. **Maintain** intelligence while using same computational resources

**Why Start Simple:**
- Model doesn't need to be good, just needs to not get worse when context extends
- Proof of concept more important than performance
- Pure Python implementation gives us full control over architecture

## Key Insight from Test 1

**GPT-2's failure doesn't invalidate the approach - it validates the need for purpose-built architectures.**

Most existing models have hard-coded context limits. We need models designed from the ground up for flexible context scaling.

---

## Current Status

✅ **Test 1 Complete** - Identified GPT-2 boundary limitations  
🔄 **Test 2 In Progress** - Building flexible context model  
⏳ **Test 3 Planned** - Compression learning validation  
⏳ **Test 4 Planned** - End-to-end context multiplication  

This is foundational research for enabling massive context AI without requiring massive computational scaling.
