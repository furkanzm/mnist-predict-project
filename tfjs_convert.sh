#!/bin/bash

# TensorFlow.js modeline dönüştürme komutu
# TensorFlow.js converter yüklü değilse önce şu komutla yüklenmeli:
# pip install tensorflowjs

tensorflowjs_converter --input_format keras \
    saved_model/mnist_ann_extended.h5 \
    tfjs_model/