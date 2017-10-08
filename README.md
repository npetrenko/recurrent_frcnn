# keras-frcnn
keras-tensorflow implementation of Faster-RCNN with recurrence layers for video detection

USAGE:
- all main configuration of training is done through editing train_frcnn.py. First it processes all videos and writes train cache to disk - it may take up a lot of space depending on your configuration.

NOTES:
- config.py contains all settings for the train or test run. The default settings match those in the original Faster-RCNN
paper. The anchor box sizes are [128, 256, 512] and the ratios are [1:1, 1:2, 2:1].
- The tensorflow backend performs a resize on the pooling region, instead of max pooling. This is much more efficient and has little impact on results.


ISSUES:

- If you get this error:
`ValueError: There is a negative shape in the graph!`    
    than update keras to the newest version

- This repo was developed using python3

- If you run out of memory, try reducing the number of ROIs that are processed simultaneously. 
