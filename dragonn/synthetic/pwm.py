from __future__ import absolute_import, division, print_function
from . import util, fileProcessing as fp
import numpy as np
import math
from collections import OrderedDict

PWM_FORMAT = util.enum(
    encodeMotifsFile="encodeMotifsFile", singlePwm="singlePwm")
DEFAULT_LETTER_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

SCORE_SEQ_MODE = util.enum(
    maxScore="maxScore", bestMatch="bestMatch", continuous="continuous", topN="topN")


def getLoadPwmArgparse():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--motifsFile", required=True)
    parser.add_argument("--pwmName", required=True)
    parser.add_argument("--pseudocountProb", type=float, default=0.0)
    return parser


def processOptions(options):
    thePwm = getSpecfiedPwmFromPwmFile(options)
    options.pwm = thePwm


def getFileNamePieceFromOptions(options):
    argsToAdd = [util.ArgumentToAdd(options.pwmName, 'pwm'), util.ArgumentToAdd(
        options.pseudocountProb, 'pcPrb')]
    toReturn = util.addArguments("", argsToAdd)
    return toReturn


def getSpecfiedPwmFromPwmFile(options):
    pwms = readPwm(fp.getFileHandle(options.motifsFile),
                   pseudocountProb=options.pseudocountProb)
    if options.pwmName not in pwms:
        raise RuntimeError("pwmName " + options.pwmName +
                           " not in " + options.motifsFile)
    pwm = pwms[options.pwmName]
    return pwm


def readPwm(fileHandle, pwmFormat=PWM_FORMAT.encodeMotifsFile, pseudocountProb=0.0):
    recordedPwms = OrderedDict()
    if (pwmFormat == PWM_FORMAT.encodeMotifsFile):
        action = getReadPwmAction_encodeMotifs(recordedPwms)
    else:
        raise RuntimeError("Unsupported pwm format: " + str(pwmFormat))
    fp.performActionOnEachLineOfFile(
        fileHandle=fileHandle, transformation=fp.trimNewline, action=action
    )
    for pwm in recordedPwms.values():
        pwm.finalise(pseudocountProb=pseudocountProb)
    return recordedPwms


class PwmScore(object):

    def __init__(self, score, posStart, posEnd):
        self.score = score
        self.posStart = posStart
        self.posEnd = posEnd

    def __str__(self):
        return str(self.score) + "\t" + str(self.posStart) + "\t" + str(self.posEnd)


class Mutation(object):
    """
        class to represent a single bp mutation in a motif sequence
    """

    def __init__(self, index, previous, new, deltaScore, parentLength=None):
        """
            index: the position within the motif of the mutation
            previous: the previous base at this position
            new: the new base at this position after the mutation
            deltaScore: change in some score corresponding to this mutation change
            parentLength: optional; length of the motif. Used for assertion checks.
        """
        self.index = index
        assert previous != new
        self.previous = previous
        self.new = new
        self.deltaScore = deltaScore
        # the length of the full sequence that self.index indexes into
        self.parentLength = parentLength

    def parentLengthAssertionCheck(self, stringArr):
        """
            checks that stringArr is consistent with parentLength if defined
        """
        assert self.parentLength is None or len(stringArr) == self.parentLength

    def revert(self, stringArr):
        """
            set the base at the position of the mutation to the unmutated value
            Modifies stringArr which is an array of characters.
        """
        self.parentLengthAssertionCheck(stringArr)
        stringArr[self.index] = self.previous

    def applyMutation(self, stringArr):
        """
            set the base at the position of the mutation to the mutated value.
            Modifies stringArr which is an array of characters.
        """
        self.parentLengthAssertionCheck(stringArr)
        assert stringArr[self.index] == self.previous
        stringArr[self.index] = self.new

BEST_HIT_MODE = util.enum(pwmProb="pwmProb", logOdds="logOdds")


class PWM(object):

    def __init__(self, name, letterToIndex=DEFAULT_LETTER_TO_INDEX, background=util.DEFAULT_BACKGROUND_FREQ):
        self.name = name
        self.letterToIndex = letterToIndex
        self.indexToLetter = dict(
            (self.letterToIndex[x], x) for x in self.letterToIndex)
        self._rows = []
        self._finalised = False
        self.setBackground(background)

    def setBackground(self, background):
        # remember to update everything that might depend on the background!
        self._background = background
        self._logBackground = dict(
            (x, math.log(self._background[x])) for x in self._background)
        if (self._finalised):
            self.updateBestLogOddsHit()

    def updateBestLogOddsHit(self):
        self.bestLogOddsHit = self.computeBestHitGivenMatrix(
            self.getLogOddsRows())

    def addRow(self, weights):
        if (len(self._rows) > 0):
            assert len(weights) == len(self._rows[0])
        self._rows.append(weights)

    def finalise(self, pseudocountProb=0.001):
        assert pseudocountProb >= 0 and pseudocountProb < 1
        # will smoothen the rows with a pseudocount...
        self._rows = np.array(self._rows)
        self._rows = self._rows * \
            (1 - pseudocountProb) + float(pseudocountProb) / len(self._rows[0])
        for row in self._rows:
            assert(abs(sum(row) - 1.0) < 0.0001)
        self._logRows = np.log(self._rows)
        self._finalised = True
        self.bestPwmHit = self.computeBestHitGivenMatrix(self._rows)
        self.updateBestLogOddsHit()
        self.pwmSize = len(self._rows)

    def getBestHit(self, bestHitMode):
        if (bestHitMode == BEST_HIT_MODE.pwmProb):
            return self.bestPwmHit
        elif (bestHitMode == BEST_HIT_MODE.logOdds):
            return self.bestLogOddsHit
        else:
            raise RuntimeError("Unsupported bestHitMode " + str(bestHitMode))

    def computeSingleBpMutationEffects(self, bestHitMode):
        if (bestHitMode == BEST_HIT_MODE.pwmProb):
            return self.computeSingleBpMutationEffectsGivenMatrix(self._rows)
        elif (bestHitMode == BEST_HIT_MODE.logOdds):
            return self.computeSingleBpMutationEffectsGivenMatrix(self._logRows)
        else:
            raise RuntimeError(
                "Unsupported best hit mode: " + str(bestHitMode))

    def computeSingleBpMutationEffectsGivenMatrix(self, matrix):
        """
            matrix is some matrix where the rows are positions and the columns represent
            a value for some base at that position, where higher values are more favourable.
            It first finds the best match according to that matrix, and then ranks possible
            deviations from that best match.
        """
        # compute the impact of particular mutations at each position, relative
        # to the best hit
        possibleMutations = []
        for rowIndex, row in enumerate(matrix):
            bestColIndex = np.argmax(row)
            scoreForBestCol = row[bestColIndex]
            letterAtBestCol = self.indexToLetter[bestColIndex]
            for colIndex, scoreForCol in enumerate(row):
                if (colIndex != bestColIndex):
                    deltaScore = scoreForCol - scoreForBestCol
                    assert deltaScore <= 0
                    letter = self.indexToLetter[colIndex]
                    possibleMutations.append(Mutation(
                        index=rowIndex, previous=letterAtBestCol, new=letter, deltaScore=deltaScore, parentLength=self.pwmSize))
        # sort the mutations
        # sorts in ascending order; biggest mutations first
        possibleMutations = sorted(
            possibleMutations, key=lambda x: x.deltaScore)
        return possibleMutations

    def computeBestHitGivenMatrix(self, matrix):
        return "".join(self.indexToLetter[x] for x in (np.argmax(matrix, axis=1)))

    def getRows(self):
        if (not self._finalised):
            raise RuntimeError("Please call finalised on " + str(self.name))
        return self._rows

    def scoreSeqAtPos(self, seq, startIdx, reverseComplement=False):
        """
            This method will score the seq at startIdx:startIdx+len(seq).
            if startIdx is < 0 or startIdx is too close to the end of the string,
            returns 0.
            This behaviour is useful when you want to generate scores at a fixed number
            of positions for a set of PWMs, some of which are longer than others.
            So at each position, for startSeq you supply pos-len(pwm)/2, and if it
            does not fit the score will automatically be 0.
        """
        endIdx = startIdx + self.pwmSize
        if (not self._finalised):
            raise RuntimeError("Please call finalised on " + str(self.name))
        assert hasattr(self, '_logRows')
        if (endIdx > len(seq) or startIdx < 0):
            return 0.0  # return 0 when indicating a segment that is too short
        score = 0
        for idx in range(startIdx, endIdx):
            if (reverseComplement):
                revIdx = (endIdx - 1) - (idx - startIdx)
            letter = util.reverseComplement(
                seq[revIdx]) if reverseComplement else seq[idx]
            if (letter not in self.letterToIndex and (letter == 'N' or letter == 'n')):
                pass  # just skip the letter
            else:
                # compute the score at this position
                score += self._logRows[idx - startIdx,
                                       self.letterToIndex[letter]] - self._logBackground[letter]
        return score

    def scoreSeq(self, seq, scoreSeqOptions):
        scoreSeqMode = scoreSeqOptions.scoreSeqMode
        reverseComplementToo = scoreSeqOptions.reverseComplementToo
        if (scoreSeqMode in [SCORE_SEQ_MODE.maxScore, SCORE_SEQ_MODE.bestMatch]):
            score = -100000000000
        elif (scoreSeqMode in [SCORE_SEQ_MODE.continuous]):
            toReturn = []
        elif (scoreSeqMode == SCORE_SEQ_MODE.topN):
            import heapq
            heap = []
            if (scoreSeqOptions.greedyTopN):
                assert len(seq) >= self.pwmSize * scoreSeqOptions.topN
        else:
            raise RuntimeError("Unsupported score seq mode: " + scoreSeqMode)

        for pos in range(0, len(seq) - self.pwmSize + 1):
            scoreHere = self.scoreSeqAtPos(seq, pos, reverseComplement=False)
            if (reverseComplementToo):
                scoreHere = max(scoreHere, self.scoreSeqAtPos(
                    seq, pos, reverseComplement=True))
            if (scoreSeqMode in [SCORE_SEQ_MODE.bestMatch, SCORE_SEQ_MODE.maxScore]):
                # Get the maximum score
                if scoreHere > score:
                    # The current score is larger than the previous largest
                    # score, so store it and the current sequence
                    score = scoreHere
                    if (scoreSeqMode in [SCORE_SEQ_MODE.bestMatch]):
                        bestMatch = seq[pos:pos + self.pwmSize]
                        bestMatchPosStart = pos
                        bestMatchPosEnd = pos + self.pwmSize
            elif (scoreSeqMode in [SCORE_SEQ_MODE.continuous]):
                toReturn.append(scoreHere)
            elif (scoreSeqMode == SCORE_SEQ_MODE.topN):
                heapq.heappush(heap, (-1 * scoreHere, scoreHere,
                                      pos, pos + self.pwmSize))  # it's a minheap
            else:
                # The current mode is not supported
                raise RuntimeError(
                    "Unsupported score seq mode: " + scoreSeqMode)

        if (scoreSeqMode in [SCORE_SEQ_MODE.maxScore]):
            return [score]
        elif (scoreSeqMode in [SCORE_SEQ_MODE.bestMatch]):
            return [bestMatch, score, bestMatchPosStart, bestMatchPosEnd]
        elif (scoreSeqMode in [SCORE_SEQ_MODE.continuous]):
            return toReturn
        elif (scoreSeqMode == SCORE_SEQ_MODE.topN):
            topNscores = []
            if (scoreSeqOptions.greedyTopN):
                occupiedHits = np.zeros(len(seq))
            while (len(topNscores) < scoreSeqOptions.topN):
                if (len(heap) > 0):
                    (negScore, score, posStart, posEnd) = heapq.heappop(heap)
                else:
                    (negScore, score, posStart, posEnd) = (0, 0, -1, -1)
                if (not scoreSeqOptions.greedyTopN or np.sum(occupiedHits[posStart:posEnd]) == 0):
                    topNscores.append(PwmScore(score, posStart, posEnd))
                    if (scoreSeqOptions.greedyTopN):
                        occupiedHits[posStart:posEnd] = 1
            return topNscores
        else:
            raise RuntimeError("Unsupported score seq mode: " + scoreSeqMode)

    def getLogOddsRows(self):
        toReturn = []
        for row in self._logRows:
            logOddsRow = []
            for (i, logVal) in enumerate(row):
                logOddsRow.append(
                    logVal - self._logBackground[self.indexToLetter[i]])
            toReturn.append(logOddsRow)
        return np.array(toReturn)

    def sampleFromPwm(self):
        if (not self._finalised):
            raise RuntimeError("Please call finalised on " + str(self.name))
        sampledLetters = []
        logOdds = 0
        for row in self._rows:
            sampledIndex = util.sampleFromProbsArr(row)
            logOdds += math.log(row[sampledIndex]) - \
                self._logBackground[self.indexToLetter[sampledIndex]]
            sampledLetters.append(
                self.indexToLetter[util.sampleFromProbsArr(row)])
        return "".join(sampledLetters), logOdds

    def __str__(self):
        return self.name + "\n" + str(self._rows)


def getReadPwmAction_encodeMotifs(recordedPwms):
    currentPwm = util.VariableWrapper(None)

    def action(inp, lineNumber):
        if (inp.startswith(">")):
            inp = inp.lstrip(">")
            inpArr = inp.split()
            motifName = inpArr[0]
            currentPwm.var = PWM(motifName)
            recordedPwms[currentPwm.var.name] = currentPwm.var
        else:
            # assume that it's a line of the pwm
            assert currentPwm.var is not None
            inpArr = inp.split()
            summaryLetter = inpArr[0]
            currentPwm.var.addRow([float(x) for x in inpArr[1:]])
    return action

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pwmFile", required=True)
    args = parser.parse_args()
    pwms = readPwm(fp.getFileHandle(args.pwmFile))
    #print("\n".join([str(x) for x in pwms.values()]));
    theSeq = "ACGATGTAGCCACTGCTGAACATCTGAATATA"
    for pwm in pwms.values():
        print(pwm)
        print(pwm.sampleFromPwm())
