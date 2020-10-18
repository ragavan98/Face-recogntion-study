install opencv:
      pip install opencv-contrib-python --upgrade
      pip install opencv-python

others:
pip install numpy
pip install pillow

procedure:
    run data_generator  (datasets nu oru floder irukkum athu delete pannidatha..athukulla tha photos save aagum)
    run trainer
    run detector
   