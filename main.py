
import word
import random
import lwa
import numpy as np
import words_model
import model
import t2_mandani_inference


input_lvs = model.input_lvs
output_lv = model.output_lv


def employee_assessment(evaluations):

    words_7 = words_model.words_7
    word_grades = []
    numerical_grades = []

    for evaluation in evaluations:
        num_assessment, word_assessment = t2_mandani_inference.process(input_lvs, output_lv, model.rule_base, evaluation)
        word_grades += [word_assessment]
        numerical_grades += [num_assessment]

    W = []
    for item in words_7['words']:
        W.append(word_grades.count(item))

    h = min(item['lmf'][-1] for item in words_7['words'].values())
    m = 50
    intervals_umf = lwa.alpha_cuts_intervals(m)
    intervals_lmf = lwa.alpha_cuts_intervals(m, h)

    res_lmf = lwa.y_lmf(intervals_lmf, words_7, W)
    res_umf = lwa.y_umf(intervals_umf, words_7, W)
    res = lwa.construct_dit2fs(np.arange(*words_7['x']), intervals_lmf, res_lmf, intervals_umf, res_umf)

    sm = []
    for title, fou in words_7['words'].items():
        sm.append((title, res.similarity_measure(word.Word(None, words_7['x'], fou['lmf'], fou['umf']))))
    result = max(sm, key=lambda item: item[1])

    return(result, word_grades)


def main():
    assessments = [tuple((random.uniform(0, 10) for _ in range(len(input_lvs)))) for _ in range(100)]
    t2_mandani_inference.preprocessing(input_lvs, output_lv)

    result, word_grades = employee_assessment(assessments)

    for i in range(len(assessments)):
        print(f'Assesment #{i+1}')
        print(f'\tProductivity - {assessments[i][0]:.2}, Communication Skills - {assessments[i][1]:.2}, Adaptability - {assessments[i][2]:.2}, Initiative - {assessments[i][3]:.2}')
        print(f'\tVerbal Grade: {word_grades[i]}')

    print()
    print(f'Result of assessment: {result[0]}')
    print(f'Numeric score: {result[1]:.2}')


if __name__ == "__main__":
    main()
