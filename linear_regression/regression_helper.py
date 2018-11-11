class RegressionHelper:

    def getRegressionEquation(self, params):
        equation = 'y = '
        predictorsCoefficients = params[1:]
        counter = 1
        predictors = [str(params[0])]
        for coef in predictorsCoefficients:
            predictor = 'x' + str(counter)
            predictor = str(coef) + '*' + predictor
            predictors.append(predictor)
            counter += 1
        equation = equation + ' + '.join(predictors, )
        return equation
