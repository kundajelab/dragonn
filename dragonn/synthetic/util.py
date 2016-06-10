from __future__ import absolute_import, division, print_function
import sys
import os
import datetime
import smtplib
import random
import glob
import json
import copy
import numpy as np
from scipy import signal
from collections import OrderedDict
from collections import namedtuple


def enum(**enums):
    class Enum(object):
        pass
    toReturn = Enum
    for key, val in enums.items():
        if hasattr(val, '__call__'):
            setattr(toReturn, key, staticmethod(val))
        else:
            setattr(toReturn, key, val)
    toReturn.vals = [x for x in enums.values()]
    toReturn.theDict = enums
    return toReturn


def normaliseEntriesByMeanAndSdev(arr):
    import numpy as np
    #assert np.mean(arr)==0 or np.mean(arr) < 10**(-7), np.mean(arr)
    return (arr - np.mean(arr)) / np.std(arr)


def normaliseEntriesBySdev(arr):
    return (arr) / np.std(arr)


def normaliseRowsByMeanAndSdev_firstFourSeq(arr):
    # normalises each row by mean and sdev but
    # treats the first four as one track type
    means = [np.mean(arr[:4])] * 4
    sdevs = [np.std(arr[:4])] * 4
    means.extend(np.mean(arr[4:], axis=1))
    sdevs.extend(np.std(arr[4:], axis=1))
    sdevs = [1 if x == 0 else x for x in sdevs]
    return (arr - np.array(means)[:, None]) / (np.array(sdevs)[:, None])


def normaliseEntriesZeroToOne(arr):
    minArr = np.min(arr)
    return (arr - minArr) / (np.max(arr) - minArr)


def divideByPerPositionRange(arr):
    assert arr.shape[0] == 4
    perPositionRange = np.max(arr, axis=0) - np.min(arr, axis=0)
    return arr / np.max(perPositionRange)


def normaliseEntriesByMeanAndTwoNorm(arr):
    theMean = np.mean(arr)
    return (arr - theMean) / np.sqrt(np.sum(np.square(arr - theMean)))


def runningWindowOneOver(func, arr, smallerArr, windowSize):
    runningWindowVals = func(arr, smallerArr, windowSize)
    runningWindowVals = np.ma.fix_invalid(runningWindowVals, copy=False)
    toMask = np.abs(runningWindowVals) == 0
    return 1.0 / (runningWindowVals + 1 * toMask)


def computeRunningWindowOneOverTwoNorm_2d(arr, smallerArr, windowSize):
    return runningWindowOneOver(computeRunningWindowTwoNorm_2d, arr, smallerArr, windowSize)


def computeRunningWindowOneOverMaxActivation_2d(arr, smallerArr, windowSize):
    return runningWindowOneOver(computeRunningWindowMaxActivation_2d, arr, smallerArr, windowSize)


def computeRunningWindowSum_2d(arr, smallerArr, windowSize):
    return np.array(computeRunningWindowSum(np.sum(arr, axis=0), windowSize))


def crossCorrelation_2d(arr, smallerArr, windowSize):
    assert len(arr.shape) == 2
    reversedSmaller = smallerArr[::-1, ::-1]
    crossCorrelations = signal.fftconvolve(arr, reversedSmaller, mode='valid')
    return crossCorrelations

CROSSC_NORMFUNC = enum(meanAndSdev=normaliseEntriesByMeanAndSdev, meanAndTwoNorm=normaliseEntriesByMeanAndTwoNorm, sdev=normaliseEntriesBySdev,
                       meanAndSdev_byRow_firstFourSeq=normaliseRowsByMeanAndSdev_firstFourSeq, none=lambda x: x, zeroToOne=normaliseEntriesZeroToOne, perPositionRange=divideByPerPositionRange)
PERPOS_NORMFUNC = enum(oneOverTwoNorm=computeRunningWindowOneOverTwoNorm_2d, oneOverMaxAct=computeRunningWindowOneOverMaxActivation_2d,
                       theSum=computeRunningWindowSum_2d, crossCorrelate=crossCorrelation_2d)


def crossCorrelateArraysLengthwise(arr1, arr2, normaliseFunc, smallerPerPosNormFuncs=[], largerPerPosNormFuncs=[], auxLargerForPerPosNorm=None, auxLargerPerPosNormFuncs=[], pad=True):
    if (len(auxLargerPerPosNormFuncs) > 0):
        assert auxLargerForPerPosNorm is not None
    assert len(arr1.shape) == 2, "arr must be 2d...did you use np.squeeze to get rid of 1-d axes? arr dims are: " + str(arr1.shape)
    assert len(arr2.shape) == 2, "arr must be 2d...did you use np.squeeze to get rid of 1-d axes? arr dims are: " + str(arr1.shape)
    # is a lengthwise correlation
    assert arr1.shape[0] == arr2.shape[0]
    normArr1 = normaliseFunc(arr1)
    normArr2 = normaliseFunc(arr2)
    # determine larger one
    if (normArr1.shape[1] > normArr2.shape[1]):
        smaller = normArr2
        larger = normArr1
        firstIsSmaller = False
    else:
        smaller = normArr1
        larger = normArr2
        firstIsSmaller = True

    normalisedSmaller = smaller
    for perPosNormFunc in smallerPerPosNormFuncs:
        normalisedSmaller = normalisedSmaller * perPosNormFunc(
            arr=smaller, smallerArr=None, windowSize=smaller.shape[1])
    smaller = normalisedSmaller

    if (pad):
        # pad the larger one
        pad_width = [(0, 0), [smaller.shape[1] - 1] * 2]
        mode = 'constant'
        paddedLarger = np.pad(larger, pad_width=pad_width, mode=mode)
        if (auxLargerForPerPosNorm is not None):
            auxLargerForPerPosNorm = np.pad(
                auxLargerForPerPosNorm, pad_width=pad_width, mode=mode)
    else:
        paddedLarger = larger
    if (auxLargerForPerPosNorm is not None):
        assert auxLargerForPerPosNorm.shape == paddedLarger.shape\
            , (paddedLarger.shape, auxLargerForPerPosNorm.shape)
    reversedSmaller = smaller[::-1, ::-1]
    crossCorrelations = signal.fftconvolve(
        paddedLarger, reversedSmaller, mode='valid')
    for perPosNormFunc in largerPerPosNormFuncs:
        crossCorrelations *= perPosNormFunc(arr=paddedLarger,
                                            smallerArr=smaller, windowSize=smaller.shape[1])
    for perPosNormFunc in auxLargerPerPosNormFuncs:
        normVals = perPosNormFunc(
            arr=auxLargerForPerPosNorm, smallerArr=smaller, windowSize=smaller.shape[1])
        assert np.isnan(
            np.sum(normVals)) == False, (auxLargerForPerPosNorm, normVals, perPosNormFunc)
        crossCorrelations *= normVals
    if (pad):
        assert crossCorrelations.shape == (
            1, larger.shape[1] + smaller.shape[1] - 1)
    else:
        assert crossCorrelations.shape == (
            1, larger.shape[1] - (smaller.shape[1] - 1))
    # the len of the smaller array will be used to calculate index
    # shifts when you do somethign like find the point of the max
    # cross correlation
    # also it's crossCorrelations[0] because we are only interested in the
    # lengthwise cross correlations; first dim has size 1.
    assert np.isnan(np.sum(crossCorrelations[0])) == False
    return crossCorrelations[0], firstIsSmaller, smaller.shape[1]


def getBestLengthwiseCrossCorrelationOfArrays(arr1, arr2, normaliseFunc, smallerPerPosNormFuncs, largerPerPosNormFuncs):
    crossCorrelations, firstIsSmaller, smallerLen = crossCorrelateArraysLengthwise(
        arr1, arr2, normaliseFunc=normaliseFunc, smallerPerPosNormFuncs=smallerPerPosNormFuncs, largerPerPosNormFuncs=largerPerPosNormFuncs)
    correlationIdx = np.argmax(crossCorrelations)
    return crossCorrelations[correlationIdx]\
        , (correlationIdx - (smallerLen - 1))\
        , firstIsSmaller


def isNumpy(obj):
    # return true if the object is a numpy object
    return type(obj).__module__ == np.__name__


def npArrayIfList(arr):
    # cast something to a numpy array if it's a list,
    # otherwise just return the original object.
    # this is to avoid making an unnecessary copy
    # if something is already a numpy array.
    if (isNumpy(arr) == False):
        return np.array(arr)
    else:
        return arr

ArgsAndKwargs = namedtuple("ArgsAndKwargs", ["args", "kwargs"])

DEFAULT_LETTER_ORDERING = ['A', 'C', 'G', 'T']
DEFAULT_BACKGROUND_FREQ = OrderedDict(
    [('A', 0.27), ('C', 0.23), ('G', 0.23), ('T', 0.27)])


class DiscreteDistribution(object):

    def __init__(self, valToFreq):
        """
            valToFreq: OrderedDict where the keys are the possible things to sample, and the values are their frequencies
        """
        self.valToFreq = valToFreq
        self.freqArr = valToFreq.values()  # array representing only the probabilities
        # map from index in freqArr to the corresponding value it represents
        self.indexToVal = dict((x[0], x[1])
                               for x in enumerate(valToFreq.keys()))
DEFAULT_BASE_DISCRETE_DISTRIBUTION = DiscreteDistribution(
    DEFAULT_BACKGROUND_FREQ)


class GetBest(object):

    def __init__(self):
        self.bestObject = None
        self.bestVal = None

    def process(self, theObject, val):
        replace = self.bestObject == None or self.isBetter(val)
        if (replace):
            self.bestObject = theObject
            self.bestVal = val
        return replace

    def isBetter(self, val):
        raise NotImplementedError()

    def getBest(self):
        return self.bestObject, self.bestVal

    def getBestVal(self):
        return self.bestVal

    def getBestObj(self):
        return self.bestObject


class GetBest_Max(GetBest):

    def isBetter(self, val):
        return val > self.bestVal


class GetBest_Min(GetBest):

    def isBetter(self, val):
        return val < self.bestVal


def isBetter(value, referenceValue, isLargerBetter):
    if (isLargerBetter):
        return value > referenceValue
    else:
        return value < referenceValue


def isBetterOrEqual(value, referenceValue, isLargerBetter):
    if (isLargerBetter):
        return value >= referenceValue
    else:
        return value <= referenceValue


def addDictionary(toUpdate, toAdd, initVal=0, mergeFunc=lambda x, y: x + y):
    """
        Defaults to addition, technically applicable any time you want to
        update a dictionary (toUpdate) with the entries of another dictionary
        (toAdd) using a particular operation (eg: adding corresponding keys)
    """
    for key in toAdd:
        if key not in toUpdate:
            toUpdate[key] = initVal
        toUpdate[key] = mergeFunc(toUpdate[key], toAdd[key])


def getExtremeN(toSort, N, keyFunc):
    """
        Returns the indices
    """
    enumeratedToSort = [x for x in enumerate(toSort)]
    sortedVals = sorted(enumeratedToSort, key=lambda x: keyFunc(x[1]))
    return [x[0] for x in sortedVals[0:N]]


reverseComplementLookup = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
                           'a': 't', 't': 'a', 'g': 'c', 'c': 'g', 'N': 'N', 'n': 'n'}


def reverseComplement(sequence):
    reversedSequence = sequence[::-1]
    reverseComplemented = "".join(
        [reverseComplementLookup[x] for x in reversedSequence])
    return reverseComplemented


def executeAsSystemCall(commandToExecute):
    print("Executing: " + commandToExecute)
    if (os.system(commandToExecute)):
        raise Exception("Error encountered with command " + commandToExecute)


def executeForAllFilesInDirectory(directory, function, fileFilterFunction=lambda x: True):
    filesInDirectory = glob.glob(directory + "/*")
    filesInDirectory = [
        aFile for aFile in filesInDirectory if fileFilterFunction(aFile)]
    for aFile in filesInDirectory:
        function(aFile)


def enum(**enums):
    class Enum(object):
        pass
    toReturn = Enum  # type('Enum', (), {})
    for key, val in enums.items():
        if hasattr(val, '__call__'):
            setattr(toReturn, key, staticmethod(val))
        else:
            setattr(toReturn, key, val)
    toReturn.vals = [x for x in enums.values()]
    toReturn.theDict = enums
    return toReturn


def combineEnums(*enums):
    newEnumDict = OrderedDict()
    for anEnum in enums:
        newEnumDict.update(anEnum.theDict)
    return enum(**newEnumDict)

SplitNames = enum(train="train", valid="valid", test="test", eval="eval")


def getTempDir():
    tempOutputDir = os.environ.get('TMP')
    if (tempOutputDir is None or tempOutputDir == ""):
        raise SystemError(
            "Please set the TMP environment variable to the temp output directory!")
    return tempOutputDir

# randomly shuffles the input arrays (correspondingly)
# mutates arrs!


def shuffleArray(*arrs):
    if len(arrs) == 0:
        raise ValueError("should supply at least one input array")
    lenOfArrs = len(arrs[0])
    # sanity check that all lengths are equal
    for arr in arrs:
        if (len(arr) != lenOfArrs):
            raise ValueError("First supplied array had length " + str(lenOfArrs) +
                             " but a subsequent array had length " + str(len(arr)))
    for i in range(0, lenOfArrs):
        # randomly select index:
        chosenIndex = random.randint(i, lenOfArrs - 1)
        for arr in arrs:
            valAtIndex = arr[chosenIndex]
            # swap
            arr[chosenIndex] = arr[i]
            arr[i] = valAtIndex
    return arrs


def sampleWithoutReplacement(arr, numToSample):
    arrayCopy = [x for x in arr]
    for i in range(numToSample):
        randomIndex = int(random.random() * (len(arrayCopy) - i)) + i
        swapIndices(arrayCopy, i, randomIndex)
    return arrayCopy[0:numToSample]


def chainFunctions(*functions):
    if (len(functions) < 2):
        raise ValueError(
            "input to chainFunctions should have at least two arguments")

    def chainedFunctions(x):
        x = functions[0](x)
        for function in functions[1:]:
            x = function(x)
        return x
    return chainedFunctions


def parseJsonFile(fileName):
    fileHandle = open(fileName)
    data = json.load(fileHandle)
    fileHandle.close()
    return data


def parseYamlFile(fileName):
    fileHandle = open(fileName)
    return parseYamlFileHandle(fileHandle)


def parseYamlFileHandle(fileHandle):
    import yaml
    data = yaml.load(fileHandle)
    fileHandle.close()
    return data


def parseMultipleYamlFiles(pathsToYaml):
    parsedYaml = {}
    for pathToYaml in pathsToYaml:
        parsedYaml.update(parseYamlFile(pathToYaml))
    return parsedYaml


def checkForAttributes(item, attributesToCheckFor, itemName=None):
    for attributeToCheckFor in attributesToCheckFor:
        if hasattr(item, attributeToCheckFor) == False:
            raise Exception("supplied item " + ("" if itemName is None else itemName) +
                            " should have attribute " + attributeToCheckFor)


def transformType(inp, theType):
    if (theType == "int"):
        return int(inp)
    elif (theType == "float"):
        return float(inp)
    elif (theType == "str"):
        return str(inp)
    else:
        raise ValueError("Unrecognised type " + theType)


class Entity(object):

    def __init__(self, id):
        self.id = id
        self.attributes = {'id': id}

    def addAttribute(self, attributeName, value):
        if (attributeName in self.attributes):
            if (self.attributes[attributeName] != value):
                raise ValueError("Attribute " + attributeName + " already exists for " + str(self.id) +
                                 " and has value " + str(self.attributes[attributeName]) + " which is not " + str(value) + "\n")
        self.attributes[attributeName] = value

    def getAttribute(self, attributeName):
        if (attributeName in self.attributes):
            return self.attributes[attributeName]
        else:
            raise ValueError("Attribute " + attributeName + " not present on entity " +
                             self.id + ". Present attributes are: " + str(self.attributes.keys()))

    def hasAttribute(self, attributeName):
        return attributeName in self.attributes


def printAttributes(entities, attributesToPrint, outputFile):
    title = "id"
    for attribute in attributesToPrint:
        title += "\t" + attribute
    f = open(outputFile, 'w')
    f.write(title + "\n")
    for entity in entities:
        line = entity.id
        for attribute in attributesToPrint:
            line += "\t" + str(entity.getAttribute(attribute))
        f.write(line + "\n")
    f.close()


def floatRange(start, end, step):
    """ Like range but for floats..."""
    val = start
    while (val <= end):
        yield val
        val += step


class VariableWrapper():
    """ For when I want reference-type access to an immutable"""

    def __init__(self, var):
        self.var = var


def getMaxIndex(arr):
    maxSoFar = arr[0]
    maxIndex = 0
    for i in range(0, len(arr)):
        if maxSoFar < arr[i]:
            maxSoFar = arr[i]
            maxIndex = i
    return maxIndex


def splitIntegerIntoProportions(integer, proportions):
    # just assign 'floor' proportion*integer to each one, and give the
    # leftovers to the largest one.
    sizesToReturn = [int(integer * prop) for prop in proportions]
    total = sum(sizesToReturn)
    leftover = integer - total
    sizesToReturn[getMaxIndex(proportions)] += leftover
    return sizesToReturn


def getDateTimeString(datetimeFormat="%y-%m-%d-%H-%M"):
    today = datetime.datetime.now()
    return datetime.datetime.strftime(today, datetimeFormat)


def sendEmails(tos, frm, subject, contents):
    for to in tos:
        sendEmail(to, frm, subject, contents)


def sendEmail(to, frm, subject, contents):
    from email.mime.text import MIMEText
    msg = MIMEText(contents)
    msg['Subject'] = subject
    msg['From'] = frm
    msg['To'] = str(to)
    s = smtplib.SMTP('smtp.stanford.edu')
    s.starttls()
    s.sendmail(frm, to, msg.as_string())
    s.quit()

# returns the string of the traceback


def getErrorTraceback():
    import traceback
    return traceback.format_exc()


def getStackTrace():
    import traceback
    return "\n".join(traceback.format_stack)


def assertMutuallyExclusiveAttributes(obj, attrs):
    arr = [presentAndNotNone(obj, attr) for attr in attrs]
    if (sum(arr) > 1):
        raise AssertionError("At most one of " + str(attrs) + " should be set")


def assertAtLeastOneSet(obj, attrs):
    if (not any([presentAndNotNone(obj, attr) for attr in attrs])):
        raise AssertionError("At least one of " +
                             str(attrs) + " should be set")


def assertLessThanOrEqual(obj, smallerAttrName, largerAttrName):
    smaller = getattr(obj, smallerAttrName)
    larger = getattr(obj, largerAttrName)
    if smaller > larger:
        raise AssertionError(smallerAttrName + " should be <= " + largerAttrName
                             + " are " + str(smaller) + " and " + str(larger) + " respectively.")


def assertAllOrNone(obj, attrNames):
    allNone = all([presentAndNotNone(obj, attr) for attr in attrNames])
    noneNone = all([absentOrNone(obj, attr) for attr in attrNames])
    if (not (allNone or noneNone)):
        raise AssertionError("Either all should be none or"
                             + " none should be none, but values are"
                             + " " + str([(attr, getattr(obj, attr)) for attr in attrNames]))


def presentAndNotNone(obj, attr):
    if (hasattr(obj, attr) and getattr(obj, attr) is not None):
        return True
    else:
        return False


def presentAndNotNoneOrFalse(obj, attr):
    if (hasattr(obj, attr) and getattr(obj, attr) is not None and getattr(obj, attr) != False):
        return True
    else:
        return False


def absentOrNone(obj, attr):
    return (presentAndNotNone(obj, attr) == False)


def assertHasAttributes(obj, attributes, explanation):
    for attr in attributes:
        if (absentOrNone(obj, attr)):
            raise AssertionError(attr, "should be set", explanation)


def assertDoesNotHaveAttributes(obj, attributes, explanation):
    for attr in attributes:
        if (presentAndNotNone(obj, attr)):
            raise AssertionError(attr, "should not be set", explanation)


def sumNumpyArrays(numpyArrays):
    import numpy as np
    arr = np.zeros(numpyArrays[0].shape)
    for i in range(0, len(numpyArrays)):
        arr += numpyArrays[i]
    return arr


def avgNumpyArrays(numpyArrays):
    theSum = sumNumpyArrays(numpyArrays)
    return theSum / float(len(numpyArrays))


def invertIndices(selectedIndices, fullSetOfIndices):
    """
        Returns all indices in fullSet but not in selected.
    """
    selectedIndicesDict = dict((x, True) for x in selectedIndices)
    return [x for x in fullSetOfIndices if x not in selectedIndicesDict]


def invertPermutation(permutation):
    inverse = [None] * len(permutation)
    for newIdx, origIdx in enumerate(permutation):
        inverse[origIdx] = newIdx
    return inverse


def valToIndexMap(arr):
    """
        A map from a value to the index at which it occurs
    """
    return dict((x[1], x[0]) for x in enumerate(arr))


def arrToDict(arr):
    """
        Turn an array into a dictionary where each value maps to '1'
        used for membership testing.
    """
    return dict((x, 1) for x in arr)


def splitChromStartEnd(chromId):
    chrom_startEnd = chromId.split(":")
    chrom = chrom_startEnd[0]
    start_end = chrom_startEnd[1].split("-")
    return (chrom, int(start_end[0]), int(start_end[1]))


def makeChromStartEnd(chrom, start, end):
    return chrom + ":" + str(start) + "-" + str(end)


def intersects(chromStartEnd1, chromStartEnd2):
    if (chromStartEnd1[0] != chromStartEnd2[0]):
        return False
    else:
        #"earlier" is the one with the earlier start coordinate.
        if (chromStartEnd1[1] < chromStartEnd2[1]):
            earlier = chromStartEnd1
            later = chromStartEnd2
        else:
            earlier = chromStartEnd2
            later = chromStartEnd1
        #"intersects" if starts before the later one ends, and ends after the later one
        # starts. Note that I am assuming 0-based start and 1-based end, a la
        # string indexing.
        if ((earlier[1] < later[2]) and (earlier[2] > later[1])):
            return True
        else:
            return False


def linecount(filename):
    import subprocess
    out = subprocess.Popen(
        ['wc', '-l', filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0]
    out = out.strip()
    print(out)
    return int(out.split(' ')[0])


def defaultTransformation():
    import fileProcessing as fp
    return chainFunctions(fp.trimNewline, fp.splitByTabs)


class ArgumentToAdd(object):
    """
        Class to append runtime arguments to a string
        to facilitate auto-generation of output file names.
    """

    def __init__(self, val, argumentName=None, argNameAndValSep="-"):
        self.val = val
        self.argumentName = argumentName
        self.argNameAndValSep = argNameAndValSep

    def argNamePrefix(self):
        return ("" if self.argumentName is None else self.argumentName + str(self.argNameAndValSep))

    def transform(self):
        string = (','.join([str(el) for el in self.val])
                  if hasattr(self.val, "__len__") else str(self.val))
        return self.argNamePrefix() + string
        # return self.argNamePrefix()+str(self.val).replace(".","p")


class BooleanArgument(ArgumentToAdd):

    def transform(self):
        assert self.val  # should be True if you're calling transformation
        return self.argumentName


class CoreFileNameArgument(ArgumentToAdd):

    def transform(self):
        import fileProcessing as fp
        return self.argNamePrefix() + fp.getCoreFileName(self.val)


class ArrArgument(ArgumentToAdd):

    def __init__(self, val, argumentName, sep="+", toStringFunc=str):
        super(ArrArgument, self).__init__(val, argumentName)
        self.sep = sep
        self.toStringFunc = toStringFunc

    def transform(self):
        return self.argNamePrefix() + self.sep.join([self.toStringFunc(x) for x in self.val])


class ArrOfFileNamesArgument(ArrArgument):

    def __init__(self, val, argumentName, sep="+"):
        import fileProcessing as fp
        super(ArrOfFileNamesArgument, self).__init__(val, argumentName,
                                                     sep, toStringFunc=lambda x: fp.getCoreFileName(x))


def addArguments(string, args, joiner="_"):
    """
        args is an array of ArgumentToAdd.
    """
    for arg in args:
        string = string + ("" if arg.val is None or arg.val is False or (hasattr(
            arg.val, "__len__") and len(arg.val) == 0) else joiner + arg.transform())
    return string


def check_pid(pid):
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def submitProcess(command):
    import subprocess
    return subprocess.Popen(command)


def overrides(interface_class):
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider


def computeConfusionMatrix(actual, predictions, labelOrdering=None):
    keySet = set()
    confusionMatrix = {}
    for i in range(0, len(actual)):
        valActual = actual[i]
        valPrediction = predictions[i]
        keySet.add(valActual)
        keySet.add(valPrediction)
        if valActual not in confusionMatrix:
            confusionMatrix[valActual] = {}
        if valPrediction not in confusionMatrix[valActual]:
            confusionMatrix[valActual][valPrediction] = 0
        confusionMatrix[valActual][valPrediction] += 1
    keys = sorted(keySet) if labelOrdering is None else labelOrdering
    #normalise and reorder
    reorderedConfusionMatrix = OrderedDict()
    for valActual in keys:
        if valActual not in confusionMatrix:
            print("Why is " + str(valActual) +
                  " in the predictions but not in the actual?")
            confusionMatrix[valActual] = {}
        reorderedConfusionMatrix[valActual] = OrderedDict()
        for valPrediction in keys:
            if valPrediction not in confusionMatrix[valActual]:
                confusionMatrix[valActual][valPrediction] = 0
            reorderedConfusionMatrix[valActual][
                valPrediction] = confusionMatrix[valActual][valPrediction]

    return reorderedConfusionMatrix


def normaliseByRowsAndColumns(theMatrix):
    """
        The matrix is as a dictionary
    """
    sumEachRow = OrderedDict()
    sumEachColumn = OrderedDict()
    for row in theMatrix:
        sumEachRow[row] = 0
        for col in theMatrix[row]:
            if col not in sumEachColumn:
                sumEachColumn[col] = 0
            sumEachRow[row] += theMatrix[row][col]
            sumEachColumn[col] += theMatrix[row][col]
    normalisedConfusionMatrix_byRow = copy.deepcopy(theMatrix)
    normalisedConfusionMatrix_byColumn = copy.deepcopy(theMatrix)
    for row in theMatrix:
        for col in theMatrix[row]:
            normalisedConfusionMatrix_byRow[row][
                col] = theMatrix[row][col] / sumEachRow[row]
            normalisedConfusionMatrix_byColumn[row][
                col] = theMatrix[row][col] / sumEachColumn[col]

    return normalisedConfusionMatrix_byRow, normalisedConfusionMatrix_byColumn, sumEachRow, sumEachColumn


def sampleFromProbsArr(arrWithProbs):
    """
        Will return a sampled index
    """
    randNum = random.random()
    cdfSoFar = 0
    for (idx, prob) in enumerate(arrWithProbs):
        cdfSoFar += prob
        if (cdfSoFar >= randNum or idx == (len(arrWithProbs) - 1)):  # need the
            # letterIdx==(len(row)-1) clause because of potential floating point errors
            # that mean arrWithProbs doesn't sum to 1
            return idx


def sampleFromDiscreteDistribution(discereteDistribution):
    """
        Expecting an instance of DiscreteDistribution
    """
    return discereteDistribution.indexToVal[sampleFromProbsArr(discereteDistribution.freqArr)]


def sampleNinstancesFromDiscreteDistribution(N, discreteDistribution):
    toReturn = []
    for i in range(N):
        toReturn.append(sampleFromDiscreteDistribution(discreteDistribution))
    return toReturn


def getFromEnum(theEnum, enumName, string):  # for yaml serialisation
    if hasattr(theEnum, string):
        return getattr(theEnum, string)
    else:
        validKeys = [x for x in dir(theEnum) if (
            not x.startswith("__")) and x != 'vals']
        raise RuntimeError("No " + str(string) +
                           " support keys for enumName are: " + str(validKeys))


class Options(object):

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


def getAllPossibleSubsets(arr):
    subsets = [[]]
    for item in arr:
        newSubsets = []
        for subset in subsets:
            newSubsets.append([item] + subset)
        subsets.extend(newSubsets)
    return subsets


class TitledMappingIterator(object):
    """
        Returns an iterator over TitledArrs for the keys in titledMapping.mapping
    """

    def __init__(self, titledMapping):
        self.titledMapping = titledMapping
        self.keysIterator = iter(titledMapping.mapping)

    def next(self):
        nextKey = self.keysIterator.next()
        return self.titledMapping.getTitledArrForKey(nextKey)


class TitledMapping(object):
    """
        When each key maps to an array, and each index in the array is associated with
            a name.
    """

    def __init__(self, titleArr, flagIfInconsistent=False):
        self.mapping = OrderedDict()  # mapping from name of a key to the values
        self.titleArr = titleArr
        self.colNameToIndex = dict((x, i)
                                   for (i, x) in enumerate(self.titleArr))
        self.rowSize = len(self.titleArr)
        self.flagIfInconsistent = flagIfInconsistent

    def keyPresenceCheck(self, key):
        """
            Throws an error if the key is absent
        """
        if (key not in self.mapping):
            raise RuntimeError(
                "Key " + str(key) + " not in mapping supported feature names are " + str(self.mapping.keys()))

    def getArrForKey(self, key):
        self.keyPresenceCheck(key)
        return self.mapping[key]

    def getTitledArrForKey(self, key):
        """
            returns an instance of util.TitledArr which has: getCol(colName) and setCol(colName)
        """
        return TitledArr(self.titleArr, self.getArrForKey(key), self.colNameToIndex)

    def addKey(self, key, arr):
        if (len(arr) != self.rowSize):
            raise RuntimeError("arr should be of size " +
                               str(self.rowSize) + " but is of size " + str(len(arr)))
        if (self.flagIfInconsistent):
            if key in self.mapping:
                if (str(self.mapping[key]) != str(arr)):
                    raise RuntimeError("Tired to add " + str(arr) + " for key " + str(
                        key) + " but " + str(self.mapping[key]) + " already present")
        self.mapping[key] = arr

    def __iter__(self):
        """
            Iterator is over instances of TitledArr!
        """
        return TitledMappingIterator(self)

    def printToFile(self, fileHandle, includeRownames=True):
        import fileProcessing as fp
        fp.writeMatrixToFile(fileHandle, self.mapping.values(), self.titleArr, [
                             x for x in self.mapping.keys()])


class Titled2DMatrix(object):
    """
        has a 2D matrix, rowNames and colNames arrays
    """

    def __init__(self, colNamesPresent=False, rowNamesPresent=False, rows=None, colNames=None, rowNames=None):
        """
            Two ways to initialise either specify rowNamesPresent/colNamesPresent and
                add the actual rows with addRow, or specify rows/colNames/rowNames
                in one shot, and colNamesPresent/rowNamesPresent will be set appropriately.
        """
        if colNames is not None or rowNames is not None:
            if (rows is None):
                raise RuntimeError(
                    "If colNames or rowNames is specified, rows should be specified")

        if colNames is not None:
            colNamesPresent = True
        if rowNames is not None:
            rowNamesPresent = True

        self.rows = [] if rows is None else rows
        self.colNamesPresent = colNamesPresent
        self.rowNamesPresent = rowNamesPresent
        self.rowNames = rowNames if rowNames is not None else (
            [] if rowNamesPresent else None)
        self.colNames = colNames if colNames is not None else (
            [] if colNamesPresent else None)

    def setColNames(self, colNames):
        assert self.colNamesPresent
        self.colNames = colNames
        self.colNameToIndex = dict((x, i)
                                   for (i, x) in enumerate(self.colNames))

    def addRow(self, arr, rowName=None):
        assert (self.rowNamesPresent) or (rowName is None)
        self.rows.append(arr)
        if (rowName is not None):
            self.rowNames.append(rowName)
        if (self.colNamesPresent):
            if (len(arr) != len(self.colNames)):
                raise RuntimeError("There are " + str(len(self.colNames)) +
                                   " column names but only " + str(len(arr)) + " columns in this row")

    def normaliseRows(self):
        self.rows = rowNormalise(np.array(self.rows))

    def printToFile(self, fileHandle):
        import fileProcessing as fp
        fp.writeMatrixToFile(fileHandle, self.rows,
                             self.colNames, self.rowNames)


class TitledArr(object):

    def __init__(self, title, arr, colNameToIndex=None):
        assert len(title) == len(arr)
        self.title = title
        self.arr = arr
        self.colNameToIndex = colNameToIndex

    def getCol(self, colName):
        assert self.colNameToIndex is not None
        return self.arr[self.colNameToIndex[colName]]

    def setCol(self, colName, value):
        assert self.colNameToIndex is not None
        self.arr[self.colNameToIndex[colName]] = value


def rowNormalise(matrix):
    import numpy as np
    rowSums = np.sum(matrix, axis=1)
    rowSums = rowSums[:, None]  # expand for axis compatibility
    return matrix / rowSums


def computeCooccurence(matrix):
    """
        matrix: rows (first dim) are examples
    """
    import numpy
    coocc = matrix.T.dot(matrix)
    return rowNormalise(coocc)


def swapIndices(arr, idx1, idx2):
    temp = arr[idx1]
    arr[idx1] = arr[idx2]
    arr[idx2] = temp


def objectFromArgsAndKwargs(classOfObject, args=[], kwargs={}):
    return classOfObject(args, kwargs)


def objectFromArgsAndKwargsFromYaml(classOfObject, yamlWithArgsAndKwargs):
    return objectFromArgsAndKwargs(classOfObject, yamlWithArgsAndKwargs['args'] if 'args' in yamlWithArgsAndKwargs else [], yamlWithArgsAndKwargs['kwargs'] if 'kwargs' in yamlWithArgsAndKwargs else '')


def arrayEquals(arr1, arr2):
    """
        compares corresponding entries in arr1 and arr2
    """
    return all((x == y) for (x, y) in zip(arr1, arr2))


def autovivisect(theDict, getThingToInitialiseWith, *keys):
    for key in keys:
        if key not in theDict:
            theDict[key] = getThingToInitialiseWith()
        theDict = theDict[key]


def setOfSeqsTo2Dimages(sequences):
    import numpy as np
    # additional index for channel
    toReturn = np.zeros((len(sequences), 1, 4, len(sequences[0])))
    for (seqIdx, sequence) in enumerate(sequences):
        seqTo2DImages_fillInArray(toReturn[seqIdx][0], sequence)
    return toReturn


def seqTo2Dimage(sequence):
    import numpy as np
    toReturn = np.zeros((1, 4, len(sequence)))
    seqTo2DImages_fillInArray(toReturn[0], sequence)
    return toReturn


def imageToSeq(image):
    import numpy as np
    letterOrdering = ['A', 'C', 'G', 'T']
    # inverts one-hot encoding
    return "".join(letterOrdering[i] for i in np.argmax(image, axis=1)[0])

# Letter as 1, other letters as 0


def seqTo2DImages_fillInArray(zerosArray, sequence):
    # zerosArray should be an array of dim 4xlen(sequence), filled with zeros.
    # will mutate zerosArray
    for (i, char) in enumerate(sequence):
        if (char == "A" or char == "a"):
            charIdx = 0
        elif (char == "C" or char == "c"):
            charIdx = 1
        elif (char == "G" or char == "g"):
            charIdx = 2
        elif (char == "T" or char == "t"):
            charIdx = 3
        elif (char == "N" or char == "n"):
            continue  # leave that pos as all 0's
        else:
            raise RuntimeError("Unsupported character: " + str(char))
        zerosArray[charIdx, i] = 1

# Letter as 3, other letters as -1
# def seqTo2DImages_fillInArray(zerosArray,sequence):
#     #zerosArray should be an array of dim 4xlen(sequence), filled with zeros.
#     #will mutate zerosArray
#     for (i,char) in enumerate(sequence):
#         if (char=="A" or char=="a"):
#             charIdx = 0
#         elif (char=="C" or char=="c"):
#             charIdx = 1
#         elif (char=="G" or char=="g"):
#             charIdx = 2
#         elif (char=="T" or char=="t"):
#             charIdx = 3
#         elif (char=="N" or char=="n"):
#             continue #leave that pos as all 0's
#         else:
#             raise RuntimeError("Unsupported character: "+str(char))
#         zerosArray[charIdx,i]=1.73
#         negIdx = [el for el in range(4) if el!=charIdx]
#         zerosArray[negIdx,i]=-0.577


def doPCAonFile(theFile):
    import sklearn.decomposition
    data = np.array(fp.read2DMatrix(fp.getFileHandle(
        theFile), colNamesPresent=True, rowNamesPresent=True, contentStartIndex=1).rows)
    pca = sklearn.decomposition.PCA()
    pca.fit(data)
    return pca


def auPRC(trueY, predictedYscores, plotFileName=None):
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import precision_recall_curve
    toReturn = average_precision_score(trueY, predictedYscores)
    if (plotFileName is not None):
        precision, recall, thresholds = precision_recall_curve(
            trueY, predictedYscores)
        plotPRC(precision, recall, toReturn, plotFileName)
    return toReturn


def plotPRC(precision, recall, auc, plotFileName):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall: AUC={0:0.2f}'.format(auc))
    plt.legend(loc="lower left")
    plt.savefig(plotFileName)


def assertAttributesHaveTheSameLengths(attributes, attributeNames):
    lens = [len(attribute) for attribute in attributes]
    if (not all([x == lens[0] for x in lens])):
        raise ValueError("all of the following attributes should have the same length: " + ", ".join(attributeNames) + "."
                         "Instead, they have lengths: " + ", ".join(str(x) for x in lens))


def splitIgnoringQuotes(string, charToSplitOn=" "):
    """
        will split on charToSplitOn, ignoring things that are in quotes
    """
    string = string.lstrip()  # strip padding whitespace on left
    toReturn = []
    thisWord = []
    lastSplitPos = 0
    inQuote = False
    for char in string:
        if (inQuote):
            if char == '"':
                inQuote = False
            else:
                thisWord.append(char)
        else:
            if char == '"':
                inQuote = True
            else:
                if char == charToSplitOn:
                    toReturn.append("".join(thisWord))
                    thisWord = []
                else:
                    thisWord.append(char)
    if len(thisWord) > 0:
        toReturn.append("".join(thisWord))
    print("Parsed arguments:", toReturn)
    return toReturn

# for those rare cases where
# you want to have some keyword
# like UNDEF


def getSingleton(name):
    class __Singleton(object):

        def __init__(self):
            pass

        def __str__(self):
            return name
    return __Singleton()

UNDEF = getSingleton("UNDEF")


def throwErrorIfUnequalSets(given, expected):
    givenSet = set(given)
    expectedSet = set(expected)
    inGivenButNotExpected = givenSet.difference(expectedSet)
    inExpectedButNotGiven = expectedSet.difference(givenSet)
    if (len(inGivenButNotExpected) > 0):
        raise RuntimeError(
            "The following were given but not expected: " + str(inGivenButNotExpected))
    if (len(inExpectedButNotGiven) > 0):
        raise RuntimeError(
            "The following were expected but not given: " + str(inExpectedButNotGiven))


def formattedJsonDump(jsonData):
    return json.dumps(jsonData, indent=4, separators=(',', ': '))

# TODO: add unit test


def roundToNearest(val, nearest):
    return round(float(val) / float(nearest)) * nearest


def sampleFromRangeWithStepSize(minVal, maxVal, stepSize, cast):
    """
        cast can be just max-min
    """
    assert maxVal >= minVal
    toReturn = roundToNearest(
        (random.random() * (maxVal - minVal)) + minVal, stepSize)
    assert toReturn >= minVal
    return cast(toReturn)


class TeeOutputStreams(object):
    """
        for piping to several output streams
    """

    def __init__(self, *streams):
        self.streams = streams
        self._closed = False

    def close(self):
        for stream in self.streams:
            stream.close()
        self._closed = True

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def writelines(self, lines):
        for stream in self.streams:
            stream.writelines(lines)

    def writeable(self):
        return True

    @property
    def closed(self):
        return self._closed

    def __del__(self):
        for stream in self.streams:
            if hasattr(stream, "__del__"):
                stream.__del__()


def redirectStdoutToString(func, logger=None, emailErrorFunc=None):
    from StringIO import StringIO
    # takes a function, and returns a wrapper which redirects
    # stdout to something else for that function
    #(the function must execute in a thread)

    def wrapperFunc(*args, **kwargs):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stringIO = StringIO()
        teeStdout = TeeOutputStreams(stringIO, sys.stdout)
        teeStderr = TeeOutputStreams(stringIO, sys.stderr)
        sys.stdout = teeStdout
        sys.stderr = teeStderr
        try:
            func(*args, **kwargs)
        except Exception as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            traceback = getErrorTraceback()
            print(traceback)
            if (logger is not None):
                logger.log(traceback)
            if (emailErrorFunc is not None):
                emailErrorFunc(traceback)
            raise e
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            stdoutContents = stringIO.getvalue()
            return stdoutContents
    return wrapperFunc


def doesNotWorkForMultithreading_redirectStdout(func, redirectedStdout):
    from StringIO import StringIO
    # takes a function, and returns a wrapper which redirects
    # stdout to something else for that function
    #(the function must execute in a thread)

    def wrapperFunc(*args, **kwargs):
        old_stdout = sys.stdout
        sys.stdout = redirectedStdout
        func(*args, **kwargs)
        sys.stdout = old_stdout

dict2str_joiner = ": "
# Can also be a dictionary with iterables as values


def dict2str(theDict, sep="\n"):
    import numpy as np
    toJoinWithSeparator = []
    for key in theDict:
        val = theDict[key]
        if (hasattr(val, '__iter__') or (type(val).__module__ == "numpy.__name__")):
            if (isinstance(val, dict)):
                stringifiedVal = "{" + ", ".join(['"{0}": [{1}]'.format(subkey, ','.join(
                    [str(ely) for ely in val[subkey]])) for subkey in val]) + "}"
            else:
                stringifiedVal = "[" + ", ".join([str(x) for x in val]) + "]"
        else:
            stringifiedVal = str(val)
        toJoinWithSeparator.append(key + dict2str_joiner + stringifiedVal)
    return sep.join(toJoinWithSeparator)


def getIntervals(minVal, numSteps, **kwargs):
    intervalsToReturn = [minVal]
    for i in range(numSteps):
        intervalsToReturn.append(getNthInterval(
            minVal=minVal, numSteps=numSteps, n=i, **kwargs))
    return intervalsToReturn


def sampleFromNumSteps(numSteps, **kwargs):
    randomIndex = int(random.random() * numSteps)
    return getNthInterval(numSteps=numSteps, n=randomIndex, **kwargs)


def getNthInterval(minVal, maxVal, numSteps, n, logarithmic, roundTo, cast):
    """
        logarithmic: boolean indicating if want log numSteps vs linear
        roundTo: can be set to None for no rounding
        cast: can be set to just lambda x: x
    """
    assert maxVal >= minVal
    diff = maxVal - minVal
    # assume n > 0 here on:
    if (logarithmic):
        scalingRatio = diff / (10.0 - 1.0)
        delta = scalingRatio * (10**(float(n) / numSteps) - 1.0)
    else:
        stepSize = diff / float(numSteps)
        delta = stepSize * n
    coreVal = minVal + delta
    if (roundTo == None):
        return cast(coreVal)
    else:
        return cast(max(min(maxVal, roundToNearest(val=minVal + delta, nearest=roundTo)), minVal))


def randomlySampleFromArr(arr):
    return arr[int(random.random() * len(arr))]


class ArgParseArgument(object):

    def __init__(self, argumentName, **kwargs):
        self.argumentName = argumentName
        self.kwargs = kwargs

    def addToParser(self, parser):
        parser.add_argument("--" + self.argumentName, **self.kwargs)


def assertIsType(instance, theClass, instanceVarName):
    if (isinstance(instance, theClass) == False):
        raise RuntimeError(instanceVarName + " should be an instance of "
                           + theClass.__name__ + " but is " + str(instance.__class__))
    return True


def augmentArgparseKwargsHelpWithDefault(**argParseKwargs):
    if 'default' in argParseKwargs:
        default = argParseKwargs['default']
        if 'help' not in argParseKwargs:
            help = ""
        else:
            help = argParseKwargs['help'] + " "
        help = help + "default " + str(default)
        argParseKwargs['help'] = help
    return argParseKwargs


def assertArrayElementsEqual(arr1, arr2, threshold=0.0):
    sumAbsDiff = sumAbsDifferences(arr1, arr2)
    if (sumAbsDifferences(arr1, arr2) < threshold):
        raise RuntimeError("Arrays are not equal"
                           " sum of abs differences is " + str(sumAbsDiff))


def sumAbsDifferences(arr1, arr2):
    import numpy as np
    return np.sum(np.abs(arr1 - arr2))


def yamlToArgsString(yamlString, subDictEnclosingChars="\""):
    import yaml
    return formatDictAsArgsString(yaml.load(yamlString), subDictEnclosingChars=subDictEnclosingChars)


def formatDictAsArgsString(theDict, subDictEnclosingChars="\""):
    args = []
    for key in theDict:
        value = theDict[key]
        if isinstance(value, bool):
            if (value == True):
                args.append("--" + key)
            else:
                pass  # assuming it's a flag
        else:
            args.append("--" + key)
            if isinstance(value, list):
                args.append(" ".join(str(x) for x in value))
            elif isinstance(value, dict):
                args.append(subDictEnclosingChars
                            + " "
                            + formatDictAsArgsString(value, subDictEnclosingChars="\\" + subDictEnclosingChars)
                            + subDictEnclosingChars
                            )
            elif isinstance(value, int) or isinstance(value, float):
                args.append(str(value))
            elif isinstance(value, str):
                args.append(value)
            else:
                raise RuntimeError(
                    "Not sure how to handle value of type " + value.__class__)
    return " ".join(args)


def getRandomString(size):
    import string
    return ''.join(random.choice(string.ascii_letters + string.digits)
                   for _ in range(size))


def readMultilineRawInput(prompt):
    from builtins import input
    sys.stdout.write(prompt)
    sentinel = ''  # ends when this string is seen
    toReturn = []
    for line in iter(input, sentinel):
        if (line.endswith("\\")):
            line = line[:-1]
        else:
            if (len(line) > 0):
                line = line + "\n"
        toReturn.append(line)
    return "".join(toReturn)


def reverse_enumerate(aList):
    for index in reversed(range(len(aList))):
        yield index, aList[index]


def enumerate_skipFirst(aList):
    for index in range(1, len(aList)):
        yield index, aList[index]


def iter_skipFirst(aList):
    for index in range(1, len(aList)):
        yield aList[index]
        yield index, aList[index]


def computeRunningWindowSum(arr, windowSize):
    import numpy as np
    # returns an array s.t. element at idx i
    # represents sum from i:i+windowSize
    runningSum = 0.0
    toReturn = []
    for idx, val in enumerate(arr):
        runningSum += val
        if idx >= (windowSize - 1):
            toReturn.append(runningSum)
            runningSum -= arr[idx - (windowSize - 1)]
    return toReturn


def computeRunningWindowOp(arr, windowSize, op):
    import numpy as np
    assert len(arr.shape) == 1, arr.shape
    toReturn = np.zeros((len(arr) - windowSize + 1,))
    for offset in range(windowSize):
        numberOfWindowsThatFit = int((len(arr) - offset) / windowSize)
        endIndex = offset + windowSize * numberOfWindowsThatFit
        reshapedArr = arr[offset:endIndex].reshape((-1, windowSize))
        valsForThisOffset = op(reshapedArr, axis=-1)
        toReturn[offset:endIndex:windowSize] = valsForThisOffset
    return toReturn


def computeRunningWindowMax(arr, windowSize):
    import numpy as np
    return computeRunningWindowOp(arr, windowSize, np.max)


def computeRunningWindowTwoNorm_2d(arr, smallerArr, windowSize):
    import numpy as np
    assert len(arr.shape) == 2
    arr = np.sum(np.square(arr), axis=0)
    return np.sqrt(np.array(computeRunningWindowSum(arr, windowSize)))


def computeRunningWindowMaxActivation_2d(arr, smallerArr, windowSize):
    import numpy as np
    assert len(arr.shape) == 2
    return np.array(computeRunningWindowSum(np.max(arr, axis=0), windowSize))


def computeRunningWindowSum_2d(arr, smallerArr, windowSize):
    import numpy as np
    return np.array(computeRunningWindowSum(np.sum(arr, axis=0), windowSize))


def crossCorrelation_2d(arr, smallerArr, windowSize):
    import numpy as np
    from scipy import signal
    assert len(arr.shape) == 2
    reversedSmaller = smallerArr[::-1, ::-1]
    crossCorrelations = signal.fftconvolve(arr, reversedSmaller, mode='valid')
    return crossCorrelations


def runningWindowOneOver(func, arr, smallerArr, windowSize):
    import numpy as np
    runningWindowVals = func(arr, smallerArr, windowSize)
    runningWindowVals = np.ma.fix_invalid(runningWindowVals, copy=False)
    toMask = np.abs(runningWindowVals) == 0
    return 1.0 / (runningWindowVals + 1 * toMask)


def computeRunningWindowOneOverMaxActivation_2d(arr, smallerArr, windowSize):
    import numpy as np
    return runningWindowOneOver(computeRunningWindowMaxActivation_2d, arr, smallerArr, windowSize)


def computeRunningWindowOneOverTwoNorm_2d(arr, smallerArr, windowSize):
    import numpy as np
    return runningWindowOneOver(computeRunningWindowTwoNorm_2d, arr, smallerArr, windowSize)


class IterableFromDict(object):

    def __init__(self, theDict, defaultVal, totalLen):
        self.theDict = theDict
        self.defaultVal = defaultVal
        self.totalLen = totalLen
        self.idx = -1

    def next(self):
        self.idx += 1
        if self.idx == self.totalLen:
            raise StopIteration()
        elif (self.idx in self.theDict):
            return self.theDict[self.idx]
        else:
            return self.defaultVal

    def __iter__(self):
        return self


class SparseArrFromDict(object):

    def __init__(self, theDict, defaultVal, totalLen):
        self.theDict = theDict
        self.defaultVal = defaultVal
        self.totalLen = totalLen

    def __iter__(self):
        return IterableFromDict(
            self.theDict, self.defaultVal, self.totalLen)

    def __getitem__(self, idx):
        assert isinstance(idx, int), "only int indexing allowed"
        if idx >= self.totalLen:
            raise IndexError("Got index "
                             + str(idx)
                             + " but max is " + str(self.totalLen))
        if idx in self.theDict:
            return self.theDict[idx]
        else:
            return self.defaultVal


def getBest(arr, getterFunc, takeMax):
    """
        Will return a tuple of the
            index and the value of the best
            as extracted by getterFunc
    """
    theBest = None
    theBestOriginalVal = None
    for originalVal in arr:
        val = getterFunc(originalVal)
        if (theBest is None):
            theBest = val
            theBestOriginalVal = originalVal
        isBetter = False
        if takeMax:
            if val > theBest:
                isBetter = True
        else:
            if val < theBest:
                isBetter = True
        if (isBetter):
            theBestOriginalVal = originalVal
            theBest = val
    return theBestOriginalVal, theBest


def multiprocessing_map_printProgress(secondsBetweenUpdates, numThreads, func, iterable):
    from multiprocessing import Pool
    import time
    p = Pool(numThreads)
    res = p.map_async(func=func, iterable=iterable)
    totalTasks = res._number_left
    p.close()
    while (True):
        if (res.ready()):
            break
        remaining = res._number_left
        time.sleep(secondsBetweenUpdates)
    return res.get()


def fracToRainbowColour(frac):
    """
        frac is a number from 0 to 1. Map to
            a 3-tuple representing a rainbow colour.
        1 -> (0, 1, 0) #green
        0.75 -> (1, 0, 1) #yellow
        0.5 -> (1, 0, 0) #red
        0.25 -> (1, 1, 0) #violet
        0 -> (0, 0, 1) #blue
    """
    unscaledTuple = (min(1, 2 * frac) - max(0, (frac - 0.5) * 2),
                     max(0, 1 - (2 * frac)), max(0, 2 * (frac - 0.5)))
    # scale so max is 1
    theMax = max(unscaledTuple)
    scaledTuple = [x / float(theMax) for x in unscaledTuple]
    return scaledTuple


def sortByLabels(arr, labels):
    """
        intended use case: sorting by cluster labels for
            plotting a heatmap
    """
    return [x[1] for x in sorted(enumerate(arr), key=lambda x: labels[x[0]])]


def printRegionIds(regionIds, labels, labelFilter, outputFile, idTransformation=lambda x: x):
    import fileProcessing as fp
    """
        intended use case: printing out the regions that corresponds
            to particular cluster labels.
    """
    rowsToPrint = [idTransformation(regionIds[i]) for i
                   in range(len(regionIds)) if labelFilter(labels[i])]
    fp.writeRowsToFile(rowsToPrint, outputFile)


def printCoordinatesForLabelSubsets(regionIds, labels, labelSetsToFilterFor, outputFilePrefix):
    """
        assumes regionIds of the form chr:start-end
        labelSetsToFilter as an iterable of iterables of the
            label you want to subset.
            Will be incorportated into the filename
        outputFile will be outputFilePrefix+"_"
                            +"-".join(str(x) for x in labelsToFilter)
    """
    idTransformation = lambda x: ("\t".join(str(x) for x in splitChromStartEnd(x))
                                  + "\t" + x)
    for labelsToFilterFor in labelSetsToFilterFor:
        outputFile = (outputFilePrefix + "_labels-")\
            + ("-".join(str(x) for x in labelsToFilterFor)) + ".txt"
        setOfLabelsToFilterFor = Set(labelsToFilterFor)
        printRegionIds(regionIds, labels=labels, labelFilter=lambda x: (
            x in setOfLabelsToFilterFor), outputFile=outputFile, idTransformation=idTransformation)
    outputFile = (outputFilePrefix + "_all.txt")
    printRegionIds(regionIds, labels=labels, labelFilter=lambda x: True,
                   outputFile=outputFile, idTransformation=idTransformation)


def normaliseEntriesByMeanAndSdev(arr):
    import numpy as np
    #assert np.mean(arr)==0 or np.mean(arr) < 10**(-7), str(np.mean(arr))+' If you are using sequence as input, be sure to mean normalize'
    return (arr - np.mean(arr)) / np.std(arr)


def normaliseEntriesByMeanAndTwoNorm(arr):
    import numpy as np
    #assert np.mean(arr)==0 or np.mean(arr) < 10**(-7), str(np.mean(arr))+' If you are using sequence as input, be sure to mean normalize else comment out this line'
    theMean = np.mean(arr)
    return (arr - theMean) / np.sqrt(np.sum(np.square(arr - theMean)))


def normaliseEntriesByTwoNorm(arr):
    import numpy as np
    return arr / np.sqrt(np.sum(np.square(arr - np.mean(arr))))


def normaliseEntriesBySdev(arr):
    import numpy as np
    return (arr) / np.std(arr)


def normaliseRowsByMeanAndSdev_firstFourSeq(arr):
    # normalises each row by mean and sdev but
    # treats the first four as one track type
    import numpy as np
    means = [np.mean(arr[:4])] * 4
    sdevs = [np.std(arr[:4])] * 4
    means.extend(np.mean(arr[4:], axis=1))
    sdevs.extend(np.std(arr[4:], axis=1))
    sdevs = [1 if x == 0 else x for x in sdevs]
    return (arr - np.array(means)[:, None]) / (np.array(sdevs)[:, None])


def normaliseRowsByMeanAndSdev(arr):
    import numpy as np
    meanRows = np.mean(arr, axis=1)
    sdevRows = np.std(arr, axis=1)
    return (arr - meanRows[:, None]) / (sdevRows[:, None])


def normaliseEntriesZeroToOne(arr):
    import numpy as np
    minArr = np.min(arr)
    return (arr - minArr) / (np.max(arr) - minArr)
