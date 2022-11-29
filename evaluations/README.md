# How to evaluate Yolov5 on ENTRON dataset
It takes quite a lot of time to generate predictions over the ~70K images of the ENTRON dataset. So the analysis of a model's performance is done in 2 step. First a script is used to run inference in parallel, then a streamlit app will help the user generate metrics.

### Step 0)
Download the weights of the model from azure VM to local machine.

### Step 1) 
Use the `evaluations/generate_predictions.py` scrip to run inference on the dataset and save the output in dst path

```shell
 python evaluations/generate_predictions.py \
    --weights ./runs/traffic_light_2020_undistorted/yolov5x6_1280_cross_prod_visible_only/weights/best.pt \
    --data ~/dataset/traffic_lights_entron_classification_v2/focal_len\=650__sensor_size_hw\=1200x1920/ \
    --dst ./evaluations/predictions/bbox/yolov5x6_1280_cross_prod_visible_only_entron_imgsize=1600_weights=best_copy.json \
    --width 1600 \
    --height 960 \
    --crop 0 0 0 1152 \
    --batch-size 18 \
    --devices 2 3 4 5 6
 ```

### Step 2)
Once predictions have been made, you can look and compare different model predictions using the Streamlit app.

```
cd evaluations/analysis_app
streamlit run main.py 
```

Inside the app, select the output to investigate.