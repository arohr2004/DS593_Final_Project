# DS593_Final_Project

MSE on DINOv2: 220.82
MSE on finetuned: 131.98

## Height Prediction App

Install the project requirements:

```bash
pip install -r requirements.txt
```

Run the Streamlit app from the repo root:

```bash
streamlit run height-app/app.py
```

The app accepts `.jpg`, `.jpeg`, and `.png` images, automatically resizes them for inference, and uses `SAM.jpg` as the default example input when no image is uploaded.
