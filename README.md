# Nigerian Currency Note Fake Detection System 🏦🔍

This project implements a software-based solution for detecting fake Nigerian currency notes using image processing techniques and machine learning. The system utilizes **MATLAB** for the algorithmic pipeline and leverages **Support Vector Machines (SVM)** and **Optical Character Recognition (OCR)** for classification and denomination recognition respectively.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Workflow Diagram](#workflow-diagram)
- [Requirements](#requirements)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Code Highlights](#code-highlights)
- [Output](#output)
- [License](#license)

---

## 📖 Project Overview

The aim of this system is to assist in identifying counterfeit Nigerian naira notes through software alone — especially useful where physical detection tools are not available. The project:
- Focuses on common denominations.
- Utilizes OCR for denomination recognition.
- Employs SVM for classification between genuine and fake notes.

---

## 🔄 Workflow Diagram

```mermaid
graph TD;
    A[Start] --> B[Collect Dataset]
    B --> C[Preprocessing: Grayscale + Resize]
    C --> D[Segmentation: Threshold + ROI]
    D --> E[Feature Extraction: OCR]
    E --> F[Classification: SVM]
    F --> G[Output: Real or Fake]
    G --> H[End]
