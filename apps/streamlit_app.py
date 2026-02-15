#!/usr/bin/env python3
"""
Streamlit Web UI for YOLOFoodCal

Usage:
    streamlit run apps/streamlit_app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detector import FoodDetector
from src.estimator import CalorieEstimator
from src.nutrition_db import NutritionDatabase
from src.portion_estimator import PortionEstimator
from src.visualizer import create_result_image


# Page configuration
st.set_page_config(
    page_title="YOLOFoodCal",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def init_estimator(model_path: str):
    """Initialize the calorie estimator (cached per model path)"""
    try:
        detector = FoodDetector(
            model_path=model_path,
            conf_threshold=0.25,
            device="auto",
            verbose=False,
        )

        # Use extended nutrition database
        nutrition_path = "data/nutrition_table_extended.json"
        if not os.path.exists(nutrition_path):
            nutrition_path = "data/nutrition_table.json"

        nutrition_db = NutritionDatabase(nutrition_path)
        portion_estimator = PortionEstimator()

        estimator = CalorieEstimator(
            detector=detector,
            nutrition_db=nutrition_db,
            portion_estimator=portion_estimator,
        )

        return estimator

    except Exception as e:
        st.error(f"Error initializing estimator: {e}")
        return None


def main():
    # Title
    st.title("")
    st.title("YOLOFoodCal")
    st.markdown("### Lightweight AI Food Detection & Calorie Estimation")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Settings")

    # Model settings
    st.sidebar.header("Model Settings")

    # Model selection - prefer food model over COCO
    model_options = {}
    if os.path.exists("models/yolo11n-food.pt"):
        model_options["Trained Food Model (42 classes)"] = "models/yolo11n-food.pt"
    if os.path.exists("yolo26n-seg.pt"):
        model_options["YOLO26 Seg (COCO 80 classes)"] = "yolo26n-seg.pt"
    if os.path.exists("yolo11n-seg.pt"):
        model_options["YOLO11n Seg (COCO 80 classes)"] = "yolo11n-seg.pt"

    if not model_options:
        model_options["YOLO26 Seg (auto-download)"] = "yolo26n-seg.pt"

    selected_model = st.sidebar.selectbox(
        "Select Model", options=list(model_options.keys()), index=0
    )

    model_path = model_options[selected_model]

    # Show model info
    st.sidebar.info(f"Model: {model_path}")

    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 0.9, 0.25, 0.05)
    use_mask = st.sidebar.checkbox("Use Segmentation Mask", value=True)

    # Display settings
    st.sidebar.header("Display Settings")
    show_masks = st.sidebar.checkbox("Show Masks", value=True)
    show_summary = st.sidebar.checkbox("Show Summary", value=True)

    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        """YOLOFoodCal is a lightweight food detection 
        and calorie estimation demo using YOLO26 
        and static nutrition data."""
    )

    # Initialize estimator (cached per model_path, threshold updated dynamically)
    with st.spinner("Initializing model..."):
        estimator = init_estimator(model_path)

    if estimator is None:
        st.error("Failed to initialize the model. Please check your installation.")
        return

    # Dynamically update thresholds (not affected by cache)
    estimator.detector.conf_threshold = conf_threshold

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Image")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"]
        )

        # Or use sample images
        use_sample = st.checkbox("Use sample image", value=False)

        sample_images = {
            "Food 1": "data/sample_images/test_01.jpg",
            "Food 2": "data/sample_images/test_02.jpg",
        }

        if use_sample:
            selected_sample = st.selectbox("Select sample", list(sample_images.keys()))
            image_path = sample_images[selected_sample]

            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                st.warning("Sample image not found. Please upload an image.")
                image = None
        elif uploaded_file is not None:
            # Convert uploaded file to image
            image = Image.open(uploaded_file)
            image = np.array(image)

            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image = None

        # Display input image
        if image is not None:
            if len(image.shape) == 3 and image.shape[2] == 3:
                display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                display_image = image

            st.image(display_image, caption="Input Image", use_container_width=True)

    # Process button
    if image is not None:
        if st.button("Detect & Estimate", type="primary"):
            with st.spinner("Processing..."):
                # Convert BGR to RGB for display
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Run estimation
                food_items = estimator.estimate(img_rgb, use_mask=use_mask)
                detections = estimator.detector.detect(img_rgb, return_masks=use_mask)

                # Get summary
                summary = estimator.get_summary(food_items)

                with col2:
                    st.subheader("Results")

                    if food_items:
                        # Create result image
                        result_img = create_result_image(
                            img_rgb,
                            detections,
                            food_items,
                            show_masks=show_masks,
                            show_summary=show_summary,
                        )

                        # Convert to RGB for display
                        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        st.image(
                            result_img,
                            caption="Detection Result",
                            use_container_width=True,
                        )

                        # Show nutrition details
                        st.markdown("### Nutrition Information")

                        # Summary cards
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric(
                                "Total Calories",
                                f"{summary['total_calories']:.0f}",
                                "kcal",
                            )
                        with c2:
                            st.metric("Protein", f"{summary['total_protein']:.1f}", "g")
                        with c3:
                            st.metric("Carbs", f"{summary['total_carbs']:.1f}", "g")
                        with c4:
                            st.metric("Fat", f"{summary['total_fat']:.1f}", "g")

                        # Food items table
                        st.markdown("### Detected Items")

                        for item in food_items:
                            with st.expander(
                                f"{item.name_en} ({item.name})", expanded=True
                            ):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.write(f"**Confidence:** {item.confidence:.2f}")
                                    st.write(f"**Portion:** {item.portion_grams:.1f}g")
                                    st.write(f"**Volume:** {item.portion_ml:.1f}ml")
                                with col_b:
                                    st.write(
                                        f"**Calories:** {item.total_calories:.1f} kcal"
                                    )
                                    st.write(f"**Protein:** {item.total_protein:.1f}g")
                                    st.write(f"**Carbs:** {item.total_carbs:.1f}g")
                                    st.write(f"**Fat:** {item.total_fat:.1f}g")
                    else:
                        st.warning("No food detected in the image.")

                        # Show original image
                        st.image(
                            img_rgb, caption="No detections", use_container_width=True
                        )


if __name__ == "__main__":
    main()
