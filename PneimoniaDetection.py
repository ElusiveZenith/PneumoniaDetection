'''
By Austin Dorsey
Started: 10/12/18
Last modified: 10/23/18
'''

import tensorflow as tf
import numpy as np
import pydicom as dicom
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import os
import math
import cv2
import time


def IoU(one, two):
    '''Calculates the Intersection over Union.
    Args:
        one: Tensor of shape (None, 4). The 4 integers are [top, bottom, left, right] top left being 0,0
        two: Tensor of same shape and format as one.
    Returns:
        IoU in a tensor of origenal first dimention (None).
    '''
    with tf.name_scope("IoU"):
        oneT = tf.transpose(one)
        twoT = tf.transpose(two)
        top = tf.maximum(oneT[0], twoT[0])
        bottom = tf.minimum(oneT[1], twoT[1])
        left = tf.maximum(oneT[2], twoT[2])
        right = tf.minimum(oneT[3], twoT[3])
        height = tf.nn.relu(tf.subtract(bottom, top))
        width = tf.nn.relu(tf.subtract(right, left))
        intersection = tf.multiply(height, width)
        areaOne = tf.multiply(tf.subtract(oneT[1], oneT[0]), tf.subtract(oneT[3], oneT[2]))
        areaTwo = tf.multiply(tf.subtract(twoT[1], twoT[0]), tf.subtract(twoT[3], twoT[2]))
        union = tf.subtract(tf.add(areaOne, areaTwo), intersection)
        iou = tf.divide(intersection, union)
    return tf.abs(tf.where(tf.is_nan(iou), tf.zeros_like(iou), iou))


def normalizeImgs(imgs):
    '''Subtracts mean and then divides by standard distribution.
    Args:
        imgs: List of images.
    Returns:
        List of images.
    '''
    imgs -= np.mean(imgs)
    imgs /= np.std(imgs)
    return imgs


def resizeImgs(imgs, size):
    '''Resizes imgs.
    Args:
        imgs: List of images to resize.
        size: Shape to resize images.
    Return:
        List of resized images.
    '''
    return np.array([resizeImg(img, size) for img in imgs])


def getCost(predict, true):
    '''Calculates the cost of predicted and true.
    Args:
        predict: 1D tensor of each 5 values are [top, bottem, left, right, accuracy]
        true: Tensor of same shape and format as predicted.
    Returns:
        Cost of predict and true.
    '''
    with tf.name_scope("cost"):
        predictNoNan = tf.where(tf.is_nan(predict), tf.zeros_like(predict), predict)
        trueNoNan = tf.where(tf.is_nan(true), tf.zeros_like(true), true)
        predictNoNanCast = tf.cast(predictNoNan, tf.float64)
        trueNoNanCast = tf.cast(trueNoNan, tf.float64)
        predictShaped = tf.reshape(predictNoNanCast, (-1, 5))
        trueShaped = tf.reshape(trueNoNanCast, (-1, 5))
        iou = IoU(predictShaped[:,:4], trueShaped[:,:4])
        rawCost = tf.abs(tf.subtract(predictShaped[:,4], iou))
        cost = tf.square(tf.scalar_mul(10.0, rawCost))
    return rawCost


def showImg(img, boxes=[], boxes2=[], threshold=0.0):
    '''Shows an image and it's bounding boxes.
    Args:
        img: ndarray of the image.
        boxes: List of all boxes [[Xs],[Ys],[Widths],[Heights],[%]]
        boxes2: List of all boxes [[Xs],[Ys],[Widths],[Heights],[%]]
        threshold: If % of the box is not above threshold, it will not be displayed.
    '''
    boxes = np.where(np.isnan(boxes), np.zeros_like(boxes), boxes)
    boxes2 = np.where(np.isnan(boxes2), np.zeros_like(boxes2), boxes2)
    fig, ax = plt.subplots(1)
    ax.imshow(img, cmap='gray')
    for i in range(len(boxes[0])):
        if boxes[4][i] < threshold:
            continue
        ax.text(int(boxes[0][i]), int(boxes[1][i]) + 1, str(boxes[4][i]), color='r')
        rect = patches.Rectangle((int(boxes[0][i]), int(boxes[1][i])), width=boxes[2][i], height=boxes[3][i], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    for i in range(len(boxes2[0])):
        if boxes2[4][i] < threshold:
            continue
        ax.text(int(boxes2[0][i]), int(boxes2[1][i]) + 1, str(boxes2[4][i]), color='g')
        rect = patches.Rectangle((int(boxes2[0][i]), int(boxes2[1][i])), width=boxes2[2][i], height=boxes2[3][i], linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    plt.show()


def saveImg(img, path, boxes=[], boxes2=[], threshold=0.0):
    '''Saves an image and it's bounding boxes.
    Args:
        img: ndarray of the image.
        boxes: List of all boxes [[Xs],[Ys],[Widths],[Heights],[%]]
        boxes2: List of all boxes [[Xs],[Ys],[Widths],[Heights],[%]]
        threshold: If % of the box is not above threshold, it will not be displayed.
    '''
    boxes = np.where(np.isnan(boxes), np.zeros_like(boxes), boxes)
    boxes2 = np.where(np.isnan(boxes2), np.zeros_like(boxes2), boxes2)
    fig, ax = plt.subplots(1)
    ax.imshow(img, cmap='gray')
    for i in range(len(boxes[0])):
        if boxes[4][i] < threshold:
            continue
        ax.text(int(boxes[0][i]), int(boxes[1][i]) + 1, str(boxes[4][i]), color='r')
        rect = patches.Rectangle((int(boxes[0][i]), int(boxes[1][i])), width=boxes[2][i], height=boxes[3][i], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    for i in range(len(boxes2[0])):
        if boxes2[4][i] < threshold:
            continue
        ax.text(int(boxes2[0][i]), int(boxes2[1][i]) + 1, str(boxes2[4][i]), color='g')
        rect = patches.Rectangle((int(boxes2[0][i]), int(boxes2[1][i])), width=boxes2[2][i], height=boxes2[3][i], linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    try:
        os.makedirs(os.path.split(path)[0])
    except FileExistsError:
        pass
    plt.savefig(path)


def loadTrainLabels(filepath):
    '''Reads csv file and converts it to a dictionary to be used.
    Args:
        filepath: Path to the csv that contain the labels.
    Returns:
        Dictionary. Key is PatientId. Values bounding box values [[Xs],[Ys],[Widths],[Heights],[Targets]]
    '''
    lists = pd.read_csv(filepath).values.tolist()
    lables = {}
    last = ''
    runningList = []
    for entry in lists:
        if entry[0] is last:
            runningList.append(entry[1:])
            continue
        else:
            lables[last] = np.array(runningList).T.tolist()
            last = entry[0]
            runningList = [entry[1:]]
    lables[last] = np.array(runningList).T.tolist()
    return lables


def getFiles(path, exclude=None, ignoreAfter='_'):
    '''Gets the filenames of the files in path excluding those in exclude after removing characturs after 
    the last instance of ignoreAfter. 
    Example file abcde_1.txt will be excluded if exclude contains abcde.txt and ignoreAfter='_'

    Args:
        valRatio: Persentage of the training data to make validation set.
    Returns:
        trainingFiles: List of the relitive file paths of training data plus augmented.
        valFiles: List of the relitive file paths of validation data.
    '''
    fullFilename = os.listdir(path)
    if exclude is None:
        return [os.path.abspath(os.path.join(path, filename)) for filename in fullFilename]
    exclude = [os.path.splitext(x)[0] for x in exclude]
    return [os.path.abspath(os.path.join(path, filename)) for filename in fullFilename if filename[:filename.rfind(ignoreAfter)] not in exclude]


def loadImg(filePath):
    '''Loads the .dcm file at filePath.
    Args:
        filePath: Path to the .dcm file to load
    Returns:
        Tuple of (PatientID, ndarray of image)
    '''
    dcm = dicom.dcmread(filePath)
    return (dcm.PatientID, dcm.pixel_array)


def loadBatch(files):
    '''Loads a list of .dcm files.
    Args:
        files: List of paths to the .dcm files to load
    Returns:
        List of tuple of (PatientID, ndarray of image)
    '''
    imgs = []
    for file in files:
        imgs.append(loadImg(file))
    return imgs


def miniBatches(x, batchSize=100):
    '''Splits it into batches of batchSize. The last batch will be what is left less than batchSize. 
    Returns lists of those batches.
    Args:
        x: List to be split.
        batchSize: Max size of batch.
    Returns:
        xBatches: List of batches of x.
    '''
    xBatches = []
    a = int(len(x))
    batches = math.ceil(a / batchSize)
    for batch in range(batches):
        xBatches.append(x[batchSize * batch : min(batchSize * (batch + 1), a)])
    return xBatches


def saveTrainValSplit(trainFiles, valFiles, path):
    '''Saves which files are being used in training and which is being used for
    validation to a .csv file, so that training can be resumed later while
    keeping validation set sepporate.
    Args:
        trainFiles: List of train file paths.
        valFiles: List of validation file paths.
        path: Path to save the file.
    '''
    filename = "TrainValFilenameSplit.csv"
    df = pd.DataFrame([trainFiles, valFiles])
    df.to_csv(os.path.join(path, filename))


def getTrainValFilenames(path, valRatio=0.0, trainFilename="training", augmentedFilename=None):
    '''Gets filenames and splits them into training and validation sets adding augmented filename
    to the training set.
    Args:
        valRatio: Persentage of the training data to make validation set.
    Returns:
        trainingFiles: List of the relitive file paths of training data plus augmented.
        valFiles: List of the relitive file paths of validation data.
    '''
    ogFilenames = getFiles(os.path.join(path, trainFilename))
    np.random.shuffle(ogFilenames)
    split = int(len(ogFilenames) * valRatio)
    valFiles, trainingFiles = ogFilenames[:split], ogFilenames[split:]
    if augmentedFilename is not None:
        trainingFiles.append(getFiles(os.path.join(path, augmentedFilename), exclude=valFiles))
    return trainingFiles, valFiles


def outToBoxes(out, imgShape, gridWidth=16, gridHeight=16):
    '''Convers the 1D output to the standard for boxes. revirce of labelListToOut()
    Args:
        out: 1D output of the model
        imgShape: Shape of the image. Use origenal shape to get back to
                  how data was befor being passed through labelListToOut()
        gridWidth: Number of grids wide that the image is diveded into.
        gridHeight: Number of grids tall that the image is diveded into.
    Returns:
        Array of boxes [[Xs],[Ys],[Widths],[Heights],[%]]
    '''
    grid = out.reshape([gridWidth, gridHeight, 5])
    gridPixWidth = imgShape[0] / gridWidth
    gridPixHeight = imgShape[1] / gridHeight
    for xIndex in range(gridWidth):
        for yIndex in range(gridHeight):
                xCenter = ((grid[xIndex][yIndex][0] / 100) + xIndex) * gridPixWidth
                yCenter = ((grid[xIndex][yIndex][1] / 100) + yIndex) * gridPixHeight
                w = grid[xIndex][yIndex][2]
                h = grid[xIndex][yIndex][3]
                x = xCenter - (h / 2.)
                y = yCenter - (w / 2.)
                per = grid[xIndex][yIndex][4] * 100
                grid[xIndex][yIndex] = [x, y, w, h, per]
    grid = np.where(np.isnan(grid), np.zeros_like(grid), grid)
    return grid.reshape([-1, 5])


def labelListToOut(entries, imgShape, ogImgShape, gridWidth=16, gridHeight=16):
    '''Converts labels into 1D array to feed for training.
    Args:
        lables: Boxes in the standard box format [[Xs],[Ys],[Widths],[Heights],[Targets]]
        imgShape: Shape of the image.
        ogImgShape: Origenal shape of the image befor any resizing.
        gridWidth: Number of grids wide that the image is diveded into.
        gridHeight: Number of grids tall that the image is diveded into.
    Return:
        out: 1D output of the model
    '''
    out = []
    for entry in entries:
        labelOut = np.zeros([gridWidth,gridHeight,1,5])
        for i in range(len(entry[0])):
            if np.isnan(entry[0][i]):
                continue
            xCenter = (entry[0][i] + (entry[2][i] / 2)) * imgShape[0] / ogImgShape[0]
            yCenter = (entry[1][i] + (entry[3][i] / 2)) * imgShape[1] / ogImgShape[1]
            gridPixWidth = imgShape[0] / gridWidth
            gridPixHeight = imgShape[1] / gridHeight
            xIndex = int(xCenter // gridPixWidth)
            yIndex = int(yCenter // gridPixHeight)
            if xIndex == gridWidth:
                xIndex -= 1
            if yIndex == gridHeight:
                yIndex -= 1
            x = (xCenter % gridPixWidth) * 100 / gridPixWidth
            y = (yCenter % gridPixHeight) * 100  / gridPixHeight
            w = entry[2][i]
            h = entry[3][i]
            labelOut[xIndex][yIndex][0] = [x, y, w, h, 100]
        out.append(labelOut.flatten())
    return np.array(out)


def prepairYForTraining(y, out):
    '''Converts the % eliment to IoU which is what is should be when slightly off.
    Args:
        y: The true value.
        out: What the model predicted. Used to calculate IoU.
    Returns:
        New y.
    '''
    shapedOut = tf.reshape(out, [-1,5])
    shapedY = tf.reshape(y, [-1,5])
    iou = IoU(shapedOut[:,:4], shapedY[:,:4])
    percent = tf.scalar_mul(100., iou)
    percentShaped = tf.reshape(percent, [-1, 1])
    newY = tf.concat([shapedY[:,:4], percentShaped], axis=1)
    return tf.reshape(newY, [-1, 1280])


def resizeImg(img, size):
    '''Resizes imgs.
    Args:
        imgs: List of images to resize.
        size: Shape to resize images.
    Return:
        List of resized images.
    '''
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)


def trainNew():
    path = "E:\Documents\Programing\Python\PneimoniaDetection\PneimoniaDetection"
    labelFilename = "stage_1_train_labels.csv"
    trainFilename = "stage_1_train_images"
    trainFiles, valFiles = getTrainValFilenames(path, 0.005, trainFilename=trainFilename)
    yFull = loadTrainLabels(os.path.join(path, labelFilename))
    imgShape = (256, 256)
    ogImgShape = (1024, 1024)
    xValData = loadBatch(valFiles)
    xValIDs = [x[0] for x in xValData]
    xValImgs = [x[1] for x in xValData]
    xValImgs = normalizeImgs(xValImgs)
    xValImgs = resizeImgs(xValImgs, imgShape)
    xValImgs = xValImgs.reshape(-1, imgShape[0], imgShape[1], 1)
    yVal = labelListToOut([yFull[id] for id in xValIDs], imgShape, ogImgShape)
    epochs = 50
    batchSize = 160
    rate = 0.0001
    xFileBatches = miniBatches(trainFiles, batchSize)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(epochs):
            for i, xFileBatch in enumerate(xFileBatches):
                xData = loadBatch(xFileBatch)
                xIDs = [x[0] for x in xData]
                xImgs = [x[1] for x in xData]
                xImgs = normalizeImgs(xImgs)
                xImgs = resizeImgs(xImgs, imgShape)
                xImgs = xImgs.reshape(-1, imgShape[0], imgShape[1], 1)
                yOut = labelListToOut([yFull[id] for id in xIDs], imgShape, ogImgShape)
                sess.run(trainingOp, feed_dict={x: xImgs, y: yOut, training: True, learningRate: rate})
            costTrain = cost.eval(feed_dict={x: xImgs, y: yOut, training: False})
            costVal = cost.eval(feed_dict={x: xValImgs, y: yVal, training: False})
            print(epoch, "Cost Training:", costTrain, "Cost Validation:", costVal)
            if epoch % 1 == 0:
                timeStamp = time.strftime("%Y%m%d-%H%M%S")
                pathChk = os.path.join(path, "checkpoints", timeStamp)
                saver.save(sess, os.path.join(pathChk, "myModel.ckpt"))
                saveTrainValSplit(trainFiles, valFiles, pathChk)
                with open(os.path.join(pathChk, "Info.txt"), 'w') as file:
                    file.write(str(timeStamp + "\n"))
                    file.write(str(str(epoch) + "\n"))
                    file.write(str("Cost Training: " + str(costTrain) + " Cost Validation: " + str(costVal) + "\n"))
                for i in range(5):
                    showChoice = xValData[np.random.choice(range(len(xValData)))]
                    choiceReshaped = resizeImg(showChoice[1], imgShape)
                    choiceReshaped = choiceReshaped.reshape(-1, imgShape[0], imgShape[1], 1)
                    predicted = out.eval(feed_dict={x: choiceReshaped, training: False})
                    predictedBoxes = outToBoxes(predicted, ogImgShape)
                    pathImg = os.path.join(pathChk, "images", str("image"+str(i)+".png"))
                    saveImg(showChoice[1], pathImg, predictedBoxes, yFull[showChoice[0]])
                    with open(os.path.join(pathChk, str("data"+str(i)+".txt")), 'w') as file:
                        print(str("Y Boxes:" + ' '.join(str(a) for a in yFull[showChoice[0]]) + '\n'))
                        for a in labelListToOut([yFull[showChoice[0]]], imgShape, ogImgShape):
                            for b in a:
                                file.write(str(str(b) + " "))
                        file.write("\n")
                        newY = prepairYForTraining(tf.constant([yFull[showChoice[0]]]), tf.constant([predicted])).eval()
                        for a in newY:
                            for b in a:
                                file.write(str(str(b) + " "))
        saver.save(sess, os.path.join(path, "checkpoints", "latest", "myModelLast.ckpt"))
        saveTrainValSplit(trainFiles, valFiles, os.path.join(path, "checkpoints", "latest"))
        with open(os.path.join(path, "checkpoints", "latest", "Info.txt")) as file:
            file.write(str("Cost Training: " + str(costTrain) + " Cost Validation: " + str(costVal) + "\n"))
        for i in range(5):
            showChoice = xValData[np.random.choice(range(len(xValData)))]
            choiceReshaped = resizeImg(showChoice[1], imgShape)
            choiceReshaped = choiceReshaped.reshape(-1, imgShape[0], imgShape[1], 1)
            predicted = out.eval(feed_dict={x: choiceReshaped, training: False})
            predictedBoxes = outToBoxes(predicted, ogImgShape)
            showImg(showChoice[1], predictedBoxes, yFull[showChoice[0]])
    return


def resumeTraining(ckptPath):
    path = "E:\Documents\Programing\Python\PneimoniaDetection\PneimoniaDetection"
    imgShape = (256, 256)
    ogImgShape = (1024, 1024)
    epochs = 11
    batchSize = 160
    rate = 0.001
    #load splits
    #load val
    #load & prep y
    #batches
    with tf.Session() as sess:
        saver.restore(sess, path)
    return


x = tf.placeholder(tf.float32, shape=(None, 256, 256, 1), name="x")
y = tf.placeholder(tf.float32, shape=(None, 16*16*5), name="y")
training = tf.placeholder(tf.bool, shape=(), name="training")
learningRate = tf.placeholder(tf.float32, shape=(), name="learningRate")


with tf.name_scope("cnn"):
    con1 = tf.layers.conv2d(x, filters=32, kernel_size=[3,3], strides=[1,1], padding="VALID")
    con2 = tf.layers.conv2d(con1, filters=64, kernel_size=[3,3], strides=[2,2], padding="VALID")
    con3 = tf.layers.conv2d(con2, filters=32, kernel_size=[1,1], strides=[1,1], padding="VALID")
    con4 = tf.layers.conv2d(con3, filters=64, kernel_size=[3,3], strides=[1,1], padding="VALID")
    con5 = tf.layers.conv2d(con4, filters=128, kernel_size=[3,3], strides=[2,2], padding="VALID")
    con6 = tf.layers.conv2d(con5, filters=64, kernel_size=[1,1], strides=[1,1], padding="VALID")
    con7 = tf.layers.conv2d(con6, filters=128, kernel_size=[3,3], strides=[1,1], padding="VALID")
    con8 = tf.layers.conv2d(con7, filters=64, kernel_size=[1,1], strides=[1,1], padding="VALID")
    con9 = tf.layers.conv2d(con8, filters=128, kernel_size=[3,3], strides=[1,1], padding="VALID")
    con10 = tf.layers.conv2d(con9, filters=256, kernel_size=[3,3], strides=[2,2], padding="VALID")
    con11 = tf.layers.conv2d(con10, filters=128, kernel_size=[1,1], strides=[1,1], padding="VALID")
    con12 = tf.layers.conv2d(con11, filters=256, kernel_size=[3,3], strides=[1,1], padding="VALID")
    con13 = tf.layers.conv2d(con12, filters=128, kernel_size=[1,1], strides=[1,1], padding="VALID")
    con14 = tf.layers.conv2d(con13, filters=256, kernel_size=[3,3], strides=[1,1], padding="VALID")
    con15 = tf.layers.conv2d(con14, filters=128, kernel_size=[1,1], strides=[1,1], padding="VALID")
    con16 = tf.layers.conv2d(con15, filters=256, kernel_size=[3,3], strides=[1,1], padding="VALID")
    con17 = tf.layers.conv2d(con16, filters=128, kernel_size=[1,1], strides=[1,1], padding="VALID")
    con18 = tf.layers.conv2d(con17, filters=256, kernel_size=[3,3], strides=[1,1], padding="VALID")
    con19 = tf.layers.conv2d(con18, filters=128, kernel_size=[1,1], strides=[1,1], padding="VALID")
    con20 = tf.layers.conv2d(con19, filters=256, kernel_size=[3,3], strides=[1,1], padding="VALID")
    con21 = tf.layers.conv2d(con20, filters=128, kernel_size=[1,1], strides=[1,1], padding="VALID")
    con22 = tf.layers.conv2d(con21, filters=64, kernel_size=[1,1], strides=[1,1], padding="VALID")
    dence = tf.layers.dense(tf.layers.flatten(con22), 1024)
    bc1 = tf.layers.batch_normalization(dence, training=training, momentum=0.95)
    bc1Act = tf.nn.relu(bc1)
    out = tf.layers.dense(bc1Act, 16*16*5)


with tf.name_scope("loss"):
    trainY = prepairYForTraining(y, out)
    sqrDif = tf.squared_difference(out, trainY)
    sqrDifNoNan = tf.abs(tf.where(tf.is_nan(sqrDif), tf.zeros_like(sqrDif), sqrDif))
    cost = tf.reduce_mean(sqrDifNoNan)


with tf.name_scope("train"):
    momentum = 0.95
    optimizer = tf.train.MomentumOptimizer(learningRate, momentum)
    trainingOp = optimizer.minimize(cost)


saver = tf.train.Saver()
