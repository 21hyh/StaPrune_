## Supplementary Material

### Anonymous Code Link

The anonymous code repository is available at https://anonymous.4open.science/r/StaPrune_-7E11/ ).

### Image QA Benchmarks

**VQAv2.** VQAv2 is a widely adopted large-scale benchmark for general visual question answering, built on MSCOCO images and open-ended natural language questions. To reduce the impact of language priors, it introduces complementary question pairs that encourage models to rely more on visual grounding rather than superficial linguistic patterns. It is commonly used to evaluate visual semantic understanding, cross-modal alignment, and open-ended reasoning ability.

**GQA.** GQA is designed to assess compositional reasoning through questions grounded in scene graphs. Its questions require multi-step inference over object attributes, spatial relations, and logical constraints, enabling systematic evaluation of reasoning consistency and interpretability. Compared with general VQA benchmarks, GQA places stronger emphasis on structured visual reasoning.

**VizWiz.** VizWiz contains real-world images captured by visually impaired users, paired with spoken questions. As a result, the images often exhibit substantial noise, such as blur, occlusion, and unconventional framing, while the questions tend to reflect spontaneous and colloquial language. This benchmark evaluates model robustness under imperfect visual conditions and its practical applicability in assistive scenarios, where resilient perception is particularly important.

**ScienceQA-IMG.** ScienceQA-IMG combines visual understanding with external domain knowledge across scientific subjects such as biology, physics, astronomy, and earth science. Answering its questions often requires integrating image content with conceptual reasoning or commonsense inference. This benchmark evaluates cross-domain generalization, where image grounding and world knowledge jointly contribute to decision making.

**TextVQA.** TextVQA focuses on the ability to detect, recognize, and semantically interpret scene text in images. Many questions require reading textual content embedded in the environment, thereby coupling OCR capability with contextual visual reasoning. It is a standard benchmark for evaluating real-world image reading ability in multimodal models.

**POPE.** POPE is designed to evaluate object hallucination, namely whether a model predicts entities that are not actually present in the image. By balancing positive, negative, and uncertain query types, it provides a rigorous test of whether model outputs are grounded in visual evidence. POPE has become a standard benchmark for assessing reliability in recent multimodal models.

**MME.** MME provides a systematic and fine-grained evaluation framework covering object recognition, attribute reasoning, counting, OCR, commonsense question answering, and cross-modal grounding. Its scoring protocol reflects foundational visual understanding and multimodal alignment ability. MME is widely used as a comprehensive reference benchmark for comparing the base competence of vision-language models.

**MMBench.** MMBench is a curated benchmark covering a diverse set of multimodal tasks, with question design emphasizing semantic clarity, single-answer correctness, and low annotation noise. It spans perception, factual understanding, relational reasoning, and multi-step inference, and provides standardized evaluation pipelines across languages. It is one of the most representative benchmarks for measuring overall VLM capability.