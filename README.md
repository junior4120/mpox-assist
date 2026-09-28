# Mpox-Assist

**AI-assisted triage and clinical decision-support prototype for suspected mpox cases**

> ⚠️ **Research prototype — not a medical device.** Mpox-Assist does not provide a definitive diagnosis and does not replace clinical assessment by a qualified healthcare professional.

## Overview

Mpox-Assist is an African-led health technology project exploring how computer vision can support healthcare workers in the early triage and orientation of suspected mpox cases from skin-lesion images.

The project is designed with resource-constrained and low-connectivity healthcare settings in mind. The longer-term objective is a lightweight solution that can support frontline health workers while keeping clinical judgment and referral decisions with qualified professionals.

## Current development stage

The project has progressed from a demonstration-only prototype to an **AI model development and evaluation stage**.

| Component | Status |
|---|---|
| FastAPI backend | ✅ Functional prototype |
| Web interface | ✅ Functional MVP |
| Image-based AI pipeline | ✅ Implemented |
| Model training on publicly available mpox image data | ✅ Completed |
| Model evaluation | ✅ Completed / ongoing improvement |
| Model comparison | ✅ MobileNetV3, Xception and DenseNet121 explored |
| Explainability / Grad-CAM | ✅ Explored |
| Clinical validation with healthcare professionals | ⏳ Not yet conducted |
| Field deployment | ⏳ Not yet conducted |
| Low-bandwidth / mobile deployment | ⏳ Next development stage |

## AI approach

The current research work uses deep-learning computer vision for image-based classification. Transfer-learning architectures including **MobileNetV3, Xception and DenseNet121** have been trained and compared as part of the model-development process.

The repository also contains evaluation artifacts and model-comparison outputs where applicable. Reported experimental results should be interpreted as research results on the available dataset and **not as evidence of clinical diagnostic performance**.

## Responsible AI

Mpox-Assist follows a human-in-the-loop approach:

- AI outputs are intended to support, not replace, healthcare professionals.
- The system should use cautious language such as *suspected* or *compatible with* rather than claiming a confirmed diagnosis.
- Health and clinical images require appropriate privacy, consent and data-governance safeguards.
- Future validation should assess robustness across skin tones, clinical presentations and imaging conditions.
- Explainability techniques such as Grad-CAM are being explored to improve transparency.
- Low-bandwidth accessibility is a planned design priority.

## Architecture

Current high-level architecture:

```
Web / future Mobile Client
          ↓
       FastAPI
          ↓
   AI Inference Layer
          ↓
  Triage / Decision Support
          ↓
 Database / supervision components
```

The architecture is evolving as the project moves from demonstration toward model validation and eventual field testing.

## Evidence and limitations

Current evidence is primarily **technical**:

- a functional web MVP;
- trained deep-learning models;
- model evaluation and comparison experiments;
- explainability experiments.

The project has **not yet undergone clinical validation or large-scale field deployment**. Publicly available image data cannot by itself establish clinical effectiveness in real-world African healthcare settings.

The next validation stage requires collaboration with healthcare professionals and institutions, access to responsibly governed and locally relevant data, expert annotation, ethical review, and field evaluation.

## Roadmap

### Next stage
1. Consolidate and document model evaluation results.
2. Improve robustness and interpretability.
3. Establish collaboration with healthcare professionals and health institutions.
4. Develop a locally relevant and professionally annotated dataset under appropriate governance.
5. Validate the solution in realistic healthcare workflows.
6. Develop lightweight mobile / low-bandwidth capabilities.

### Longer-term objective
Progress from a research MVP toward a responsibly validated AI-assisted triage and referral-support tool for frontline healthcare settings.

## Disclaimer

Mpox-Assist is a research and innovation project. It is **not clinically validated, is not a substitute for professional medical assessment, and should not be used to make autonomous clinical decisions**.

## Repository structure

The repository contains the backend/API, machine-learning components, web application, tests, data documentation and model-evaluation artifacts.

For technical details, please inspect the relevant directories and experiment documentation in this repository.
