# FittingroomAnywhere_v2
Fittingroom Anywhere Demo and Final code


# project folder overview
PROJECT_DIR = "."
- main.py
- mrcnn
  - tshirt.py
  - other .py files
  - logs (mrcnn saved model)
    - mrcnn_tshirt.h5 (weight)
- cyclegan
  - model.py
  - load_data.py
  - other .py files
- datasets
  - white2stripe
    - input
      - user_input.jpg / .png
    - segmented
      - segmented.png
    - fake_output (gan generated output)
      - segmented_fake.png
    - output (final rendered output)
      - output.png / .jpg
    - model (cyclegan saved model)
      - G_A2B.json (model)
      - G_A2B.hdf5 (weight)
