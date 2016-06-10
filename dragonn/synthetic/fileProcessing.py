from __future__ import absolute_import, division, print_function
import re
import os
from . import util
import gzip


def assertParameterNecessaryForMode(parameterName, parameter, modeName, mode):
    if (parameter is None):
        raise RuntimeError(
            parameterName + " is necessary when " + modeName + " is " + mode)


def assertParameterIrrelevantForMode(parameterName, parameter, modeName, mode):
    if (parameter is not None):
        raise RuntimeError(
            parameterName + " is irrelevant when " + modeName + " is " + mode)


def unsupportedValueForMode(modeName, mode):
    raise RuntimeError("Unsupported value for " + modeName + ": " + str(mode))


def getCoreFileName(fileName):
    return getFileNameParts(fileName).coreFileName


def getFileNameParts(fileName):
    p = re.compile(r"^(.*/)?([^\./]+)(\.[^/]*)?$")
    m = p.search(fileName)
    return FileNameParts(m.group(1), m.group(2), m.group(3))


class FileNameParts:

    def __init__(self, directory, coreFileName, extension):
        self.directory = directory if (directory is not None) else os.getcwd()
        self.coreFileName = coreFileName
        self.extension = extension

    def getFullPath(self):
        return self.directory + "/" + self.fileName

    def getCoreFileNameAndExtension(self):
        return self.coreFileName + self.extension

    def getCoreFileNameWithTransformation(self, transformation=lambda x: x):
        return transformation(self.coreFileName)

    def getFileNameWithTransformation(self, transformation, extension=None):
        toReturn = self.getCoreFileNameWithTransformation(transformation)
        if (extension is not None):
            toReturn = toReturn + extension
        else:
            if (self.extension is not None):
                toReturn = toReturn + self.extension
        return toReturn

    def getFilePathWithTransformation(self, transformation=lambda x: x, extension=None):
        return self.directory + "/" + self.getFileNameWithTransformation(transformation, extension=extension)


def getFileHandle(filename, mode="r"):
    if (re.search('.gz$', filename) or re.search('.gzip', filename)):
        if (mode == "r"):
            mode = "rt"
        elif (mode == "w"):
            # I think write will actually append if the file already
            # exists...so you want to remove it if it exists
            import os.path
            if os.path.isfile(filename):
                os.remove(filename)
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)

# returns an array of all filter variables.


def splitLinesIntoOtherFiles(fileHandle, preprocessingStep, filterVariableFromLine, outputFilePathFromFilterVariable):
    filterVariablesToReturn = []
    filterVariableToOutputFileHandle = {}
    for line in fileHandle:
        processedLine = line
        if (preprocessingStep is not None):
            processedLine = preprocessingStep(processedLine)
        filterVariable = filterVariableFromLine(processedLine)
        if (filterVariable not in filterVariableToOutputFileHandle):
            outputFilePath = outputFilePathFromFilterVariable(filterVariable)
            filterVariablesToReturn.append(filterVariable)
            outputFileHandle = getFileHandle(outputFilePath, 'w')
            filterVariableToOutputFileHandle[filterVariable] = outputFileHandle
        outputFileHandle = filterVariableToOutputFileHandle[filterVariable]
        outputFileHandle.write(line)
    for fileHandle in filterVariableToOutputFileHandle.items():
        fileHandle[1].close()
    return filterVariablesToReturn

# transformation has a specified default so that this can be used to, for
# instance, unzip a gzipped file.


def transformFile(
    # should be some function of the line and the line number
    # processing to be applied before filterFunction AND transformation
    fileHandle, outputFile, transformation=lambda x: x, outputTitleFromInputTitle=None, ignoreInputTitle=False, filterFunction=None, preprocessing=None, progressUpdate=None, progressUpdateFileName=None
):

    outputFileHandle = getFileHandle(outputFile, 'w')
    i = 0
    action = lambda inp, i: outputFileHandle.write(inp)
    for line in fileHandle:
        i += 1
        if (i == 1):
            if (outputTitleFromInputTitle is not None):
                outputFileHandle.write(outputTitleFromInputTitle(line))
        processLine(line, i, ignoreInputTitle, preprocessing,
                    filterFunction, transformation, action)
        printProgress(progressUpdate, i, progressUpdateFileName)
    outputFileHandle.close()

# reades a line of the file on-demand.


class FileReader:

    def __init__(self, fileHandle, preprocessing=None, filterFunction=None, transformation=lambda x: x, ignoreInputTitle=False):
        self.fileHandle = fileHandle
        self.preprocessing = preprocessing
        self.filterFunction = filterFunction
        self.transformation = transformation
        self.ignoreInputTitle = ignoreInputTitle
        self.i = 0
        self.eof = False

    def getNextLine(self):
        line = self.fileHandle.readline()
        if (line != ""):  # empty string is eof...
            self.i += 1
            if (self.i == 1):
                if (self.ignoreInputTitle == True):
                    self.title = line
                    return self.getNextLine()

            def action(x, i):  # to be passed into processLine
                self.toReturn = x

            processLine(
                line=line, i=self.i, ignoreInputTitle=self.ignoreInputTitle, preprocessing=self.preprocessing, filterFunction=self.filterFunction, transformation=self.transformation, action=action)

            return self.toReturn
        else:
            self.eof = True
            return None


def writeToFile(outputFile, contents):
    outputFileHandle = getFileHandle(outputFile, 'w')
    writeToFileHandle(outputFileHandle, contents)


def writeToFileHandle(outputFileHandle, contents):
    outputFileHandle.write(contents)
    outputFileHandle.close()


def transformFileIntoArray(fileHandle):
    return readRowsIntoArr(fileHandle)


def performActionOnEachLineOfFile(fileHandle, action=None  # should be a function that accepts the preprocessed/filtered line and the line number
                                  # the preprocessing step is performed before
                                  # both 'filterFunction' and 'transformation'.
                                  # Originally I just had 'transformation'.
                                  , transformation=lambda x: x, ignoreInputTitle=False, filterFunction=None, preprocessing=None, actionFromTitle=None, progressUpdate=None, progressUpdateFileName=None
                                  ):
    if (actionFromTitle is None and action is None):
        raise ValueError("One of actionFromTitle or action should not be None")
    if (actionFromTitle is not None and action is not None):
        raise ValueError(
            "Only one of actionFromTitle or action can be non-None")
    if (actionFromTitle is not None and ignoreInputTitle == False):
        raise ValueError(
            "If actionFromTitle is not None, ignoreInputTitle should probably be True because it implies a title is present")

    i = 0
    for line in fileHandle:
        i += 1
        if (i == 1 and actionFromTitle is not None):
            action = actionFromTitle(line)
        processLine(line, i, ignoreInputTitle, preprocessing,
                    filterFunction, transformation, action, progressUpdate)
        printProgress(progressUpdate, i, progressUpdateFileName)

    fileHandle.close()


def performActionInBatchesOnEachLineOfFile(fileHandle, batchSize, actionOnLineInBatch, actionAtEndOfBatch, transformation=lambda x: x, filterFunction=None, preprocessing=None, progressUpdate=None, ignoreInputTitle=False):
    def action(inp, lineNumber):
        actionOnLineInBatch(inp, lineNumber)
        if (lineNumber % batchSize == 0):
            actionAtEndOfBatch()
    performActionOnEachLineOfFile(
        fileHandle=fileHandle, action=action, transformation=transformation, filterFunction=filterFunction, preprocessing=preprocessing, progressUpdate=progressUpdate, ignoreInputTitle=ignoreInputTitle
    )
    actionAtEndOfBatch()


def processLine(line, i, ignoreInputTitle, preprocessing, filterFunction, transformation, action, progressUpdate=None):
    if (i > 1 or (ignoreInputTitle == False)):
        if (preprocessing is not None):
            line = preprocessing(line)
        if (filterFunction is None or filterFunction(line, i)):
            action(transformation(line), i)


def printProgress(progressUpdate, i, fileName=None):
    if progressUpdate is not None:
        if (i % progressUpdate == 0):
            print("Processed " + str(i) + " lines" +
                  str("" if fileName is None else " of " + fileName))


def defaultTabSeppd(s):
    s = trimNewline(s)
    s = splitByTabs(s)
    return s


def defaultWhitespaceSeppd(s):
    s = trimNewline(s)
    s = s.split()
    return s


def trimNewline(s):
    return s.rstrip('\r\n')


def appendNewline(s):
    return s + "\n"  # aargh O(n) aargh FIXME if you can


def splitByDelimiter(s, delimiter):
    return s.split(delimiter)


def splitByTabs(s):
    return splitByDelimiter(s, "\t")


def stringToFloat(s):
    return float(s)


def stringToInt(s):
    return int(s)


def lambdaMaker_getAtPosition(index):
    return (lambda x: x[index])


def lambdaMaker_insertSuffixIntoFileName(suffix, separator):
    return lambda fileName: getFileNameParts(fileName).getFileNameWithTransformation(
        lambda coreFileName: coreFileName + separator + suffix
    )


def lambdaMaker_insertPrefixIntoFileName(prefix, separator):
    return lambda fileName: getFileNameParts(fileName).getFileNameWithTransformation(
        lambda coreFileName: prefix + separator + coreFileName
    )


def simpleDictionaryFromFile(fileHandle, keyIndex=0, valIndex=1, titlePresent=False, transformation=defaultTabSeppd):
    from collections import OrderedDict
    toReturn = OrderedDict()

    def action(inp, lineNumber):
        toReturn[inp[keyIndex]] = inp[valIndex]
    performActionOnEachLineOfFile(
        fileHandle=fileHandle, action=action, transformation=transformation, ignoreInputTitle=titlePresent
    )
    return toReturn

'''Accepts a title, uses it to produce a function that generates a dictionary given an array'''


def lambdaMaker_dictionaryFromLine(title, delimiter="\t"):
    lineToArr = util.chainFunctions(
        trimNewline, lambda x: splitByDelimiter(x, delimiter))
    splitTitle = lineToArr(title)

    def lineToDictionary(line):
        toReturn = {}
        for entry in enumerate(lineToArr(line)):
            toReturn[splitTitle[entry[0]]] = entry[1]
        return toReturn
    return lineToDictionary

# wrapper for the cat command


def concatenateFiles(outputFile, arrOfFilesToConcatenate):
    util.executeAsSystemCall(
        "cat " + (" ".join(arrOfFilesToConcatenate)) + " > " + outputFile)


def concatenateFiles_preprocess(
    # x is the input line, y is the transformed input file name (see
    # inputFileNameTransformation)
    # the function that transforms the path of the input file
    # function that takes input title and transforms into output title.
    # Considers title of first file in line.
    # really unfortunately named. Should have been named 'titlePresent'
    outputFile, arrOfFilesToConcatenate, transformation=lambda x, y: x, inputFileTransformation=lambda x: getFileNameParts(x).coreFileName, outputTitleFromInputTitle=None, ignoreInputTitle=False
):
    inputTitle = None
    outputFileHandle = getFileHandle(outputFile, 'w')
    for aFile in arrOfFilesToConcatenate:
        transformedInputFilename = inputFileTransformation(aFile)
        aFileHandle = getFileHandle(aFile)
        i = 0
        for line in aFileHandle:
            i += 1
            if (i == 1):
                if (outputTitleFromInputTitle is not None):
                    if (inputTitle is None):
                        inputTitle = line
                        outputFileHandle.write(
                            outputTitleFromInputTitle(inputTitle))
            if (i > 1 or (ignoreInputTitle == False)):
                outputFileHandle.write(transformation(
                    line, transformedInputFilename))
    outputFileHandle.close()


def readRowsIntoArr(fileHandle, progressUpdate=None, titlePresent=False):
    arr = []

    def action(inp, lineNumber):
        if progressUpdate is not None:
            if (lineNumber % progressUpdate == 0):
                print("processed " + str(lineNumber) + " lines")
        arr.append(inp)
    performActionOnEachLineOfFile(
        fileHandle, transformation=trimNewline, action=action, ignoreInputTitle=titlePresent
    )
    return arr


def writeRowsToFile(rows, theFile):
    writeRowsToFileHandle(rows, getFileHandle(theFile, 'w'))


def writeRowsToFileHandle(rows, fileHandle):
    for row in rows:
        fileHandle.write(str(row) + "\n")
    fileHandle.close()


def readColIntoArr(fileHandle, col=0, titlePresent=True):
    arr = []

    def action(inp, lineNumber):
        arr.append(inp[col])
    performActionOnEachLineOfFile(
        fileHandle, transformation=defaultTabSeppd, action=action, ignoreInputTitle=titlePresent
    )
    return arr


def read2DMatrix(fileHandle, colNamesPresent=False, rowNamesPresent=False, contentType=float, contentStartIndex=None, contentEndIndex=None, progressUpdate=None, numpify=False):
    """
        "numpify" will return a numpy mat for the rows
        returns an instance of util.Titled2DMatrix
        Has attributes rows, rowNames, colNames
    """
    fileHandle = getFileHandle(fileHandle) if isinstance(
        fileHandle, str) else fileHandle
    if (contentStartIndex is None):
        contentStartIndex = 1 if rowNamesPresent else 0
    if (contentEndIndex is not None):
        assert contentEndIndex > contentStartIndex
    titled2DMatrix = util.Titled2DMatrix(
        colNamesPresent=colNamesPresent, rowNamesPresent=rowNamesPresent)
    contentEndIndexWrapper = util.VariableWrapper(None)

    def action(inp, lineNumber):
        if (lineNumber == 1 and colNamesPresent):
            contentEndIndexWrapper.var = len(
                inp) if contentEndIndex is None else contentEndIndex
            titled2DMatrix.setColNames(
                inp[contentStartIndex:contentEndIndexWrapper.var])
            assert contentEndIndexWrapper.var > contentStartIndex
        else:
            rowName = inp[0] if rowNamesPresent else None
            # ignore the column denoting the name of the row
            arr = [contentType(x) for x in inp[
                contentStartIndex:contentEndIndexWrapper.var]]
            titled2DMatrix.addRow(arr, rowName=rowName)
    performActionOnEachLineOfFile(
        fileHandle=fileHandle, action=action, transformation=defaultTabSeppd, ignoreInputTitle=False, progressUpdate=progressUpdate
    )
    if (numpify):
        import numpy as np
        titled2DMatrix.rows = np.array(titled2DMatrix.rows)
    return titled2DMatrix


class SubsetOfColumnsToUseOptions(object):

    def __init__(self, mode='setOfColumnNames', columnNames=None, N=None):
        self.mode = mode
        self.columnNames = columnNames
        self.N = N
        self.integrityChecks()

    def integrityChecks(self):
        if (self.mode == 'setOfColumnNames'):
            assertParameterIrrelevantForMode(
                "N", self.N, "subsetOfColumnsToUseMode", self.mode)
            assertParameterNecessaryForMode(
                "columnNames", self.columnNames, "subsetOfColumnsToUseMode", self.mode)
        elif (self.mode == 'topN'):
            assertParameterIrrelevantForMode(
                "columnNames", self.columnNames, "subsetOfColumnsToUseMode", self.mode)
            assertParameterNecessaryForMode(
                "N", self.N, "subsetOfColumnsToUseMode", self.mode)
        else:
            # unsupportedValueForMode(modeName, mode)
            raise ValueError()


def getCoreTitledMappingAction(subsetOfColumnsToUseOptions, contentType, contentStartIndex, subsetOfRowsToUse=None, keyColumns=[0]):
    subsetOfRowsToUseMembershipDict = dict(
        (x, 1) for x in subsetOfRowsToUse) if subsetOfRowsToUse is not None else None
    indicesToCareAboutWrapper = util.VariableWrapper(None)

    def titledMappingAction(inp, lineNumber):
        if (lineNumber == 1):  # handling of the title
            if subsetOfColumnsToUseOptions is None:
                columnOrdering = inp[contentStartIndex:]
            else:
                if (subsetOfColumnsToUseOptions.mode == SubsetOfColumnsToUseMode.setOfColumnNames):
                    columnOrdering = subsetOfColumnsToUseOptions.columnNames
                elif (subsetOfColumnsToUseOptions.mode == SubsetOfColumnsToUseMode.topN):
                    columnOrdering = inp[
                        contentStartIndex:contentStartIndex + SubsetOfColumnsToUseMode.topN]
                else:
                    raise RuntimeError(
                        "Unsupported subsetOfColumnsToUseOptions.mode: " + str(subsetOfColumnsToUseOptions.mode))
                print("Subset of labels to use is specified")
                indicesLookup = dict((x, i) for (i, x) in enumerate(inp))
                indicesToCareAboutWrapper.var = []
                for labelToUse in columnOrdering:
                    indicesToCareAboutWrapper.var.append(
                        indicesLookup[labelToUse])
            return columnOrdering
        else:
            # regular line processing
            key = "_".join(inp[x] for x in keyColumns)
            if (subsetOfRowsToUseMembershipDict is None or (key in subsetOfRowsToUseMembershipDict)):
                if (indicesToCareAboutWrapper.var is None):
                    arrToAdd = [contentType(x)
                                for x in inp[contentStartIndex:]]
                else:
                    arrToAdd = [contentType(inp[x])
                                for x in indicesToCareAboutWrapper.var]
                return key, arrToAdd
            return None
    return titledMappingAction


def readTitledMapping(fileHandle, contentType=float, contentStartIndex=1, subsetOfColumnsToUseOptions=None, subsetOfRowsToUse=None, progressUpdate=None, keyColumns=[0]):
    """
        returns an instance of util.TitledMapping.
            util.TitledMapping has functions:
                - getTitledArrForKey(key): returns an instance of util.TitledArr which has: getCol(colName) and setCol(colName)
                - getArrForKey(key): returns the array for the key
                - keyPresenceCheck(key): throws an error if the key is absent
                Is also iterable! Returns an iterator of util.TitledArr
        subsetOfColumnsToUseOptions: instance of SubsetOfColumnsToUseOptions
        subsetOfRowsToUse: something that has a subset of row ids to be considered
    """

    titledMappingWrapper = util.VariableWrapper(None)
    coreTitledMappingAction = getCoreTitledMappingAction(subsetOfColumnsToUseOptions=subsetOfColumnsToUseOptions,
                                                         contentType=contentType, contentStartIndex=contentStartIndex, subsetOfRowsToUse=subsetOfRowsToUse, keyColumns=keyColumns)

    def action(inp, lineNumber):
        if (lineNumber == 1):  # handling of the title
            columnOrdering = coreTitledMappingAction(inp, lineNumber)
            titledMappingWrapper.var = util.TitledMapping(columnOrdering)
        else:
            key, arrToAdd = coreTitledMappingAction(inp, lineNumber)
            if (arrToAdd is not None):
                titledMappingWrapper.var.addKey(key, arrToAdd)
    performActionOnEachLineOfFile(
        fileHandle, transformation=defaultTabSeppd, action=action
    )
    return titledMappingWrapper.var


def writeMatrixToFile(fileHandle, rows, colNames=None, rowNames=None):
    if (colNames is not None):
        fileHandle.write(
            ("rowName\t" if rowNames is not None else "") + "\t".join(colNames) + "\n")
    for i, row in enumerate(rows):
        if (rowNames is not None):
            fileHandle.write(rowNames[i] + "\t")
        stringifiedRow = [str(x) for x in row]
        toWrite = "\t".join(stringifiedRow) + "\n"
        fileHandle.write(toWrite)
    fileHandle.close()

# will trim the newline for you


def titleColumnToIndex(title, sep="\t"):
    title = trimNewline(title)
    title = title.split(sep)
    return util.valToIndexMap(title)


def peekAtFirstLineOfFile(fileName):
    fh = getFileHandle(fileName)
    line = fh.readline()
    fh.close()
    return line


def getTitleOfFile(fileName):
    title = defaultTabSeppd(peekAtFirstLineOfFile(fileName))
    return title


class FastaIterator(object):
    """
        Returns an iterator over lines of a fasta file - assumes each sequence
        spans only one line!
    """

    def __init__(self, fileHandle, progressUpdate=None, progressUpdateFileName=None):
        self.fileHandle = fileHandle
        self.progressUpdate = progressUpdate
        self.progressUpdateFileName = progressUpdateFileName
        self.lineCount = 0

    def __iter__(self):
        return self

    def next(self):
        self.lineCount += 1
        printProgress(self.progressUpdate, self.lineCount,
                      self.progressUpdateFileName)
        # should raise StopIteration if at end of lines
        keyLine = trimNewline(self.fileHandle.next())
        sequence = trimNewline(self.fileHandle.next())
        if (keyLine.startswith(">") == False):
            raise RuntimeError(
                "Expecting a record name line that begins with > but got " + str(keyLine))
        key = keyLine.lstrip(">")
        return key, sequence


class BackupForWriteFileHandle(object):
    """
        Wrapper around a filehandle that
            backs up the file while writing,
            then deletes the backup when close
            is called
    """

    def __init__(self, fileName):
        self.fileName = fileName
        self.backupFileName = fileName + ".backup"
        os.system("cp " + self.fileName + " " + self.backupFileName)
        self.outputFileHandle = getFileHandle(self.fileName, 'w')

    def write(self, *args, **kwargs):
        self.outputFileHandle.write(*args, **kwargs)

    def close(self):
        self.outputFileHandle.close()
        os.system("rm " + self.backupFileName)

    def restore(self):
        os.system("cp " + self.backupFileName + " " + self.fileName)
        os.system("rm " + self.backupFileName)
        self.outputFileHandle.close()
