# DL Project Final Combined

ICT 4442 sentiment classification project. The four member folders contain their independently developed code and resources. The reports are at the repository root:

- [Phase 1 project synopsis](Phase_1_Project_Synopsis.docx)
- [Phase 2 interim report](Phase_2_Interim_Report.docx)
- [Final report](ICT_4442_Final_Report.docx)

| Member | Model | Source repository | Snapshot |
| --- | --- | --- | --- |
| Navneet | DistilBERT | [RA5C4L/DL-Proj](https://github.com/RA5C4L/DL-Proj) | `7456cfb` |
| Rohit | TextCNN | [rohitmosh/Rohit_230953168_DLProject](https://github.com/rohitmosh/Rohit_230953168_DLProject) | `5bbaa8d` |
| Varaang | TF-IDF MLP | [Varaang-Tiwari/230953378_DL_Project](https://github.com/Varaang-Tiwari/230953378_DL_Project) | `46346ee` |
| Arshabrata | BiLSTM | [GreatCoder69/DL_Mini_Project](https://github.com/GreatCoder69/DL_Mini_Project) | `0ee728d` |

## Repository structure

- `Navneet/`: DistilBERT implementation, experiment protocol, manifests, per-example predictions and recorded metrics.
- `Rohit/`: TextCNN implementation, training summary, audit, plots and recorded metrics.
- `Varaang/`: TF-IDF MLP implementation, split artifact, vectorizer, predictions, plots and metrics.
- `Arshabrata/`: BiLSTM implementation, shared-suite code, requirements and Colab runbook.

Each member folder retains its own dependencies and run instructions. The independently recorded results use differing model selection, transfer data and perturbation protocols; consult the final report for interpretation. A harmonized four-model comparison should use frozen shared splits and identical evaluation inputs.
