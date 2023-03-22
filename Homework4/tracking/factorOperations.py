# factorOperations.py
# -------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from typing import List, ValuesView
from bayesNet import Factor
import functools
from util import raiseNotDefined


def joinFactorsByVariableWithCallTracking(callTrackingList=None):
    def joinFactorsByVariable(factors: List[Factor], joinVariable: str):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin = [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len(
            [factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print("Factors failed joinFactorsByVariable typecheck: ", factors)
            raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +
                             "joinVariable: " + str(joinVariable) + "\n" +
                             ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))

        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable


joinFactorsByVariable = joinFactorsByVariableWithCallTracking()


########### ########### ###########
########### QUESTION 2  ###########
########### ########### ###########

def joinFactors(factors: ValuesView[Factor]):
    """
    factors: can iterate over it as if it was a list, and convert to a list.
    
    You should calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input 
    (such as getProbability and setProbability) can handle 
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factors failed joinFactors typecheck: ", factors)
            raise ValueError("unconditionedVariables can only appear in one factor. \n"
                             + "unconditionedVariables: " + str(intersect) +
                             "\nappear in more than one input factor.\n" +
                             "Input factors: \n" +
                             "\n".join(map(str, factors)))

    "*** YOUR CODE HERE ***"
    # raiseNotDefined()
    # 从前往后两两factor合并
    factors = list(factors)
    new_factor = factors[0]
    for i in range(1, len(factors)):
        # 合并后的非条件变量
        unconditionedVar = set.union(new_factor.unconditionedVariables(),
                                     factors[i].unconditionedVariables())
        conditionedVar = set()  # 合并后的条件变量

        for var in set.union(new_factor.conditionedVariables(),
                             factors[i].conditionedVariables()):
            if var not in unconditionedVar:
                conditionedVar.add(var)
        # 变量值域字典
        varDomainsDict = {**new_factor.variableDomainsDict(),
                          **factors[i].variableDomainsDict()}
        # 合并后的factor
        tmp_factor = Factor(unconditionedVar, conditionedVar, varDomainsDict)
        # 计算合并后的概率
        for assignment in tmp_factor.getAllPossibleAssignmentDicts():
            assignment1 = {}  # 前一个factor的assignment
            assignment2 = {}  # 后一个factor的assignment
            for var in set.union(new_factor.unconditionedVariables(),
                                 new_factor.conditionedVariables()):
                assignment1[var] = assignment[var]
            for var in set.union(factors[i].unconditionedVariables(),
                                 factors[i].conditionedVariables()):
                assignment2[var] = assignment[var]
            prob = new_factor.getProbability(assignment1) * \
                   factors[i].getProbability(assignment2)
            tmp_factor.setProbability(assignment, prob)
        new_factor = tmp_factor
    "*** END YOUR CODE HERE ***"
    return new_factor


########### ########### ###########
########### QUESTION 3  ###########
########### ########### ###########

def eliminateWithCallTracking(callTrackingList=None):
    def eliminate(factor: Factor, eliminationVariable: str):
        """
        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.
        
        You should calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Elimination variable is not an unconditioned variable " \
                             + "in this factor\n" +
                             "eliminationVariable: " + str(eliminationVariable) + \
                             "\nunconditionedVariables:" + str(factor.unconditionedVariables()))

        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Factor has only one unconditioned variable, so you " \
                             + "can't eliminate \nthat variable.\n" + \
                             "eliminationVariable:" + str(eliminationVariable) + "\n" + \
                             "unconditionedVariables: " + str(factor.unconditionedVariables()))

        "*** YOUR CODE HERE ***"
        # raiseNotDefined()
        # 移除一个变量后的非条件变量
        unconditionedVar = factor.unconditionedVariables()
        unconditionedVar.remove(eliminationVariable)
        # 移除一个变量后的条件变量
        conditionedVar = factor.conditionedVariables()
        # 变量值域字典
        varDomainsDict = factor.variableDomainsDict()
        # 新的factor
        new_factor = Factor(unconditionedVar, conditionedVar, varDomainsDict)
        # 计算移除一个变量后的概率
        for assignment in new_factor.getAllPossibleAssignmentDicts():
            new_factor.setProbability(assignment, 0.)
        for assignment in factor.getAllPossibleAssignmentDicts():
            prob = factor.getProbability(assignment)
            assignment.pop(eliminationVariable)
            new_factor.setProbability(assignment, new_factor.getProbability(assignment)+prob)
        "*** END YOUR CODE HERE ***"
        return new_factor

    return eliminate


eliminate = eliminateWithCallTracking()
