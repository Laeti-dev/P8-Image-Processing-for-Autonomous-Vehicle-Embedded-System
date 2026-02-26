# TODO Checklist - Image Segmentation Project

## 📋 Phase 1: Preparation (Week 1)

### Environment
- [X] Create Python virtual environment (uv)
- [X] Install dependencies (Keras, TensorFlow, FastAPI, Streamlit, etc.)
- [X] Create `requirements.txt`
- [X] Create folder structure

### Azure Setup
- [X] Create Azure for Students account
- [X] Configure Azure CLI
- [X] Create Resource Group
- [X] Explore free deployment options

### Dataset
- [X] Download leftImg8bit images
- [X] Download gtFine annotations
- [X] Verify file structure

---

## 🤖 Phase 2: Model (Week 2-3)

### Exploration
- [X] Data exploration notebook
- [X] Visualize class distribution
- [X] Create utility loading functions

### Data Generator
- [X] Implement Keras Sequence
- [X] Convert 34 classes → 8 categories
- [X] Add data augmentation (Albumentations)
- [X] Test the generator

### Architecture
- [X] Choose architecture (U-Net recommended)
- [X] Implement the model
- [X] Configure hyperparameters
- [X] Create custom metrics (Dice, IoU)

### Training
- [X] Split train/val
- [X] Configure callbacks
- [X] Create training notebook
- [X] Add visualization during training
- [X] Train without augmentation
- [X] Train with augmentation
- [X] Compare results
- [X] Save final model

### Evaluation
- [X] Calculate metrics (IoU per class)
- [X] Confusion matriX
- [X] Visualize predictions
- [ ] Document results

---

## 🔌 Phase 3: API (Week 4)

### Local Development
- [X] Create `api/app.py`
- [X] Implement model loading
- [X] Create `/predict` endpoint
- [X] Create `/health` endpoint
- [X] Error handling
- [X] Test locally

### Azure Deployment
- [ ] Create `requirements.txt` for API
- [ ] Create Azure App Service (F1 free tier)
- [ ] Deploy API
- [ ] Upload model
- [ ] Test deployed API
- [ ] Document URL

---

## 🌐 Phase 4: Web Application (Week 5)

### Local Development
- [ ] Choose Flask or Streamlit
- [ ] Create `web_app/app.py`
- [ ] List image IDs
- [ ] Image selection
- [ ] API call
- [ ] Display image + real mask + predicted mask
- [ ] Style interface
- [ ] Test locally

### Azure Deployment
- [ ] Create second App Service
- [ ] Deploy application
- [ ] Configure environment variables
- [ ] Test deployed app

---

## 📝 Phase 5: Documentation (Week 6)

### Technical Note
- [ ] Introduction and context
- [ ] State of the art
- [ ] Dataset presentation
- [ ] Model architecture
- [ ] Training pipeline
- [ ] Results and metrics
- [ ] Data augmentation impact
- [ ] Conclusion and improvements
- [ ] Graphs and visualizations

### Presentation
- [ ] Maximum 30 slides
- [ ] Introduction
- [ ] Problem statement
- [ ] State of the art
- [ ] Methodology
- [ ] Results
- [ ] Conclusion
- [ ] Visuals and diagrams

### Code Documentation
- [ ] Docstrings on all functions
- [ ] Complete README.md
- [ ] Comments on complex code

---

## ✅ Phase 6: Final Tests (Week 7)

### Tests
- [ ] Test API with different images
- [ ] Test web app end-to-end
- [ ] Verify error handling
- [ ] Test performance
- [ ] Verify Azure functionality

### Optimizations
- [ ] Optimize model size
- [ ] Optimize loading times
- [ ] Verify Azure resource usage

### Demo Preparation
- [ ] Prepare test images
- [ ] Test complete scenario
- [ ] Prepare answers to questions

---

## 🎯 Final Checklist

- [ ] Trained and saved model
- [ ] Complete training notebook
- [ ] Deployed and functional API
- [ ] Deployed and functional web application
- [ ] Complete technical note
- [ ] Presentation support
- [ ] Updated README.md
- [ ] Documented code
- [ ] All tests passed

---

**Overall Progress: 0%** ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜
