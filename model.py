
input_lvs = [

    {
        'X': (0, 10.1, 0.1),
        'name': 'Productivity',
        'terms': {
			'Limited': {'umf': ('trapmf', 0, 0, 0.55, 4.61), 'lmf': ('trapmf', 0, 0, 0.09, 1.15, 1)},
			'Basic': {'umf': ('trapmf', 0.42, 2.25, 4., 5.41), 'lmf': ('trapmf', 2.79, 3.21, 3.21, 3.71, 0.34)},
			'Moderate': {'umf': ('trapmf', 1.59, 2.75, 4.35, 6.26), 'lmf': ('trapmf', 2.79, 3.34, 3.34, 3.67, 0.35)},
			'High': {'umf': ('trapmf', 3.38, 5.5, 7.25, 9.02), 'lmf': ('trapmf', 5.79, 6.28, 6.28, 6.67, 0.33)},
			'Very-High': {'umf': ('trapmf', 4.59, 5.9, 7.25, 8.5), 'lmf': ('trapmf', 6.29, 6.67, 6.67, 7.17, 0.39)},
			'Exceptional': {'umf': ('trapmf', 7.37, 9.36, 10, 10), 'lmf': ('trapmf', 8.68, 9.91, 10, 10, 1)},
		}
    },

    {
        'X': (0, 10.1, 0.1),
        'name': 'Communication Skills',
        'terms': {
			'Basic': {'umf': ('trapmf', 0, 0, 0.55, 4.61), 'lmf': ('trapmf', 0, 0, 0.09, 1.15, 1)},
            'Effective': {'umf': ('trapmf', 0.42, 2.25, 4.00, 5.41), 'lmf': ('trapmf', 2.79, 3.21, 3.21, 0.34, 3.71)},
            'Advanced': {'umf': ('trapmf', 3.38, 5.50, 7.25, 9.02), 'lmf': ('trapmf', 5.79, 6.28, 6.28, 0.33, 6.67)},
            'Exceptional': {'umf': ('trapmf', 7.37, 9.36, 10, 10), 'lmf': ('trapmf', 8.68, 9.91, 10, 10, 1)},
		}
    },

    {
        'X': (0, 10.1, 0.1),
        'name': 'Adaptability',
        'terms': {
			'Low': {'umf': ('trapmf', 0, 0, 0.55, 4.61), 'lmf': ('trapmf', 0, 0, 0.09, 1.15, 1)},
            'Satisfactory': {'umf': ('trapmf', 0.42, 2.25, 4.00, 5.41), 'lmf': ('trapmf', 2.79, 3.21, 3.21, 0.34, 3.71)},
            'Above-Average': {'umf': ('trapmf', 3.38, 5.50, 7.25, 9.02), 'lmf': ('trapmf', 5.79, 6.28, 6.28, 0.33, 6.67)},
            'High': {'umf': ('trapmf', 7.37, 9.36, 10, 10), 'lmf': ('trapmf', 8.68, 9.91, 10, 10, 1)},
		}
    },

    {
        'X': (0, 10.1, 0.1),
        'name': 'Initiative',
        'terms': {
			'Limited': {'umf': ('trapmf', 0, 0, 0.55, 4.61), 'lmf': ('trapmf', 0, 0, 0.09, 1.15, 1)},
            'Basic': {'umf': ('trapmf', 0.42, 2.25, 4.00, 5.41), 'lmf': ('trapmf', 2.79, 3.21, 3.21, 0.34, 3.71)},
            'Proactive': {'umf': ('trapmf', 3.38, 5.50, 7.25, 9.02), 'lmf': ('trapmf', 5.79, 6.28, 6.28, 0.33, 6.67)},
            'Exceptional': {'umf': ('trapmf', 7.37, 9.36, 10, 10), 'lmf': ('trapmf', 8.68, 9.91, 10, 10, 1)},
		}
    },

]


output_lv = {
    'X': (0, 10.1, 0.1),
    'name': 'Employee Assessment',
    'terms': {
        'Unsatisfactory': {'umf': ('trapmf', 0, 0, 0.59, 3.95), 'lmf': ('trapmf', 0, 0, 0.09, 1.32, 1)},
        'Novice': {'umf': ('trapmf', 0.28, 2.00, 3.00, 5.22), 'lmf': ('trapmf', 1.79, 2.37, 2.37, 2.71, 0.48)},
        'Competent': {'umf': ('trapmf', 0.98, 2.75, 4.00, 5.41), 'lmf': ('trapmf', 2.79, 3.30, 3.30, 3.71, 0.42)},
        'Proficient': {'umf': ('trapmf', 2.38, 4.50, 6.00, 8.18), 'lmf': ('trapmf', 4.79, 5.12, 5.12, 5.35, 0.27)},
        'Advanced': {'umf': ('trapmf', 4.02, 5.65, 7.00, 8.41), 'lmf': ('trapmf', 5.89, 6.34, 6.34, 6.81, 0.40)},
        'Exceptional': {'umf': ('trapmf', 4.38, 6.50, 7.75, 9.62), 'lmf': ('trapmf', 6.79, 7.25, 7.25, 7.91, 0.47)},
        'Distinguished': {'umf': ('trapmf', 5.21, 8.27, 10, 10), 'lmf': ('trapmf', 7.66, 9.82, 10, 10, 1)},
    }
}


rule_base = [

    (('Limited', 'Basic', 'Low', 'Limited'), 'Unsatisfactory'),
    (('Limited', 'Basic', 'Low', 'Basic'), 'Unsatisfactory'),
    (('Limited', 'Basic', 'Low', 'Proactive'), 'Novice'),
    (('Limited', 'Basic', 'Low', 'Exceptional'), 'Competent'),
    (('Limited', 'Basic', 'Satisfactory', 'Limited'), 'Unsatisfactory'),
    (('Limited', 'Basic', 'Satisfactory', 'Basic'), 'Novice'),
    (('Limited', 'Basic', 'Satisfactory', 'Proactive'), 'Competent'),
    (('Limited', 'Basic', 'Satisfactory', 'Exceptional'), 'Proficient'),
    (('Limited', 'Basic', 'Above-Average', 'Limited'), 'Novice'),
    (('Limited', 'Basic', 'Above-Average', 'Basic'), 'Novice'),
    (('Limited', 'Basic', 'Above-Average', 'Proactive'), 'Competent'),
    (('Limited', 'Basic', 'Above-Average', 'Exceptional'), 'Proficient'),
    (('Limited', 'Basic', 'High', 'Limited'), 'Novice'),
    (('Limited', 'Basic', 'High', 'Basic'), 'Competent'),
    (('Limited', 'Basic', 'High', 'Proactive'), 'Proficient'),
    (('Limited', 'Basic', 'High', 'Exceptional'), 'Proficient'),
    (('Limited', 'Effective', 'Low', 'Limited'), 'Unsatisfactory'),
    (('Limited', 'Effective', 'Low', 'Basic'), 'Novice'),
    (('Limited', 'Effective', 'Low', 'Proactive'), 'Novice'),
    (('Limited', 'Effective', 'Low', 'Exceptional'), 'Competent'),
    (('Limited', 'Effective', 'Satisfactory', 'Limited'), 'Novice'),
    (('Limited', 'Effective', 'Satisfactory', 'Basic'), 'Novice'),
    (('Limited', 'Effective', 'Satisfactory', 'Proactive'), 'Competent'),
    (('Limited', 'Effective', 'Satisfactory', 'Exceptional'), 'Proficient'),
    (('Limited', 'Effective', 'Above-Average', 'Limited'), 'Novice'),
    (('Limited', 'Effective', 'Above-Average', 'Basic'), 'Competent'),
    (('Limited', 'Effective', 'Above-Average', 'Proactive'), 'Proficient'),
    (('Limited', 'Effective', 'Above-Average', 'Exceptional'), 'Proficient'),
    (('Limited', 'Effective', 'High', 'Limited'), 'Novice'),
    (('Limited', 'Effective', 'High', 'Basic'), 'Competent'),
    (('Limited', 'Effective', 'High', 'Proactive'), 'Proficient'),
    (('Limited', 'Effective', 'High', 'Exceptional'), 'Proficient'),
    (('Limited', 'Advanced', 'Low', 'Limited'), 'Unsatisfactory'),
    (('Limited', 'Advanced', 'Low', 'Basic'), 'Novice'),
    (('Limited', 'Advanced', 'Low', 'Proactive'), 'Competent'),
    (('Limited', 'Advanced', 'Low', 'Exceptional'), 'Proficient'),
    (('Limited', 'Advanced', 'Satisfactory', 'Limited'), 'Novice'),
    (('Limited', 'Advanced', 'Satisfactory', 'Basic'), 'Novice'),
    (('Limited', 'Advanced', 'Satisfactory', 'Proactive'), 'Competent'),
    (('Limited', 'Advanced', 'Satisfactory', 'Exceptional'), 'Proficient'),
    (('Limited', 'Advanced', 'Above-Average', 'Limited'), 'Novice'),
    (('Limited', 'Advanced', 'Above-Average', 'Basic'), 'Competent'),
    (('Limited', 'Advanced', 'Above-Average', 'Proactive'), 'Proficient'),
    (('Limited', 'Advanced', 'Above-Average', 'Exceptional'), 'Proficient'),
    (('Limited', 'Advanced', 'High', 'Limited'), 'Competent'),
    (('Limited', 'Advanced', 'High', 'Basic'), 'Proficient'),
    (('Limited', 'Advanced', 'High', 'Proactive'), 'Proficient'),
    (('Limited', 'Advanced', 'High', 'Exceptional'), 'Proficient'),
    (('Limited', 'Exceptional', 'Low', 'Limited'), 'Novice'),
    (('Limited', 'Exceptional', 'Low', 'Basic'), 'Novice'),
    (('Limited', 'Exceptional', 'Low', 'Proactive'), 'Competent'),
    (('Limited', 'Exceptional', 'Low', 'Exceptional'), 'Proficient'),
    (('Limited', 'Exceptional', 'Satisfactory', 'Limited'), 'Novice'),
    (('Limited', 'Exceptional', 'Satisfactory', 'Basic'), 'Competent'),
    (('Limited', 'Exceptional', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('Limited', 'Exceptional', 'Satisfactory', 'Exceptional'), 'Proficient'),
    (('Limited', 'Exceptional', 'Above-Average', 'Limited'), 'Novice'),
    (('Limited', 'Exceptional', 'Above-Average', 'Basic'), 'Competent'),
    (('Limited', 'Exceptional', 'Above-Average', 'Proactive'), 'Proficient'),
    (('Limited', 'Exceptional', 'Above-Average', 'Exceptional'), 'Proficient'),
    (('Limited', 'Exceptional', 'High', 'Limited'), 'Competent'),
    (('Limited', 'Exceptional', 'High', 'Basic'), 'Proficient'),
    (('Limited', 'Exceptional', 'High', 'Proactive'), 'Proficient'),
    (('Limited', 'Exceptional', 'High', 'Exceptional'), 'Proficient'),
    (('Basic', 'Basic', 'Low', 'Limited'), 'Unsatisfactory'),
    (('Basic', 'Basic', 'Low', 'Basic'), 'Novice'),
    (('Basic', 'Basic', 'Low', 'Proactive'), 'Competent'),
    (('Basic', 'Basic', 'Low', 'Exceptional'), 'Proficient'),
    (('Basic', 'Basic', 'Satisfactory', 'Limited'), 'Novice'),
    (('Basic', 'Basic', 'Satisfactory', 'Basic'), 'Novice'),
    (('Basic', 'Basic', 'Satisfactory', 'Proactive'), 'Competent'),
    (('Basic', 'Basic', 'Satisfactory', 'Exceptional'), 'Proficient'),
    (('Basic', 'Basic', 'Above-Average', 'Limited'), 'Novice'),
    (('Basic', 'Basic', 'Above-Average', 'Basic'), 'Competent'),
    (('Basic', 'Basic', 'Above-Average', 'Proactive'), 'Proficient'),
    (('Basic', 'Basic', 'Above-Average', 'Exceptional'), 'Proficient'),
    (('Basic', 'Basic', 'High', 'Limited'), 'Competent'),
    (('Basic', 'Basic', 'High', 'Basic'), 'Proficient'),
    (('Basic', 'Basic', 'High', 'Proactive'), 'Proficient'),
    (('Basic', 'Basic', 'High', 'Exceptional'), 'Proficient'),
    (('Basic', 'Effective', 'Low', 'Limited'), 'Novice'),
    (('Basic', 'Effective', 'Low', 'Basic'), 'Novice'),
    (('Basic', 'Effective', 'Low', 'Proactive'), 'Competent'),
    (('Basic', 'Effective', 'Low', 'Exceptional'), 'Proficient'),
    (('Basic', 'Effective', 'Satisfactory', 'Limited'), 'Novice'),
    (('Basic', 'Effective', 'Satisfactory', 'Basic'), 'Competent'),
    (('Basic', 'Effective', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('Basic', 'Effective', 'Satisfactory', 'Exceptional'), 'Proficient'),
    (('Basic', 'Effective', 'Above-Average', 'Limited'), 'Novice'),
    (('Basic', 'Effective', 'Above-Average', 'Basic'), 'Competent'),
    (('Basic', 'Effective', 'Above-Average', 'Proactive'), 'Proficient'),
    (('Basic', 'Effective', 'Above-Average', 'Exceptional'), 'Proficient'),
    (('Basic', 'Effective', 'High', 'Limited'), 'Competent'),
    (('Basic', 'Effective', 'High', 'Basic'), 'Proficient'),
    (('Basic', 'Effective', 'High', 'Proactive'), 'Proficient'),
    (('Basic', 'Effective', 'High', 'Exceptional'), 'Proficient'),
    (('Basic', 'Advanced', 'Low', 'Limited'), 'Novice'),
    (('Basic', 'Advanced', 'Low', 'Basic'), 'Novice'),
    (('Basic', 'Advanced', 'Low', 'Proactive'), 'Competent'),
    (('Basic', 'Advanced', 'Low', 'Exceptional'), 'Proficient'),
    (('Basic', 'Advanced', 'Satisfactory', 'Limited'), 'Novice'),
    (('Basic', 'Advanced', 'Satisfactory', 'Basic'), 'Competent'),
    (('Basic', 'Advanced', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('Basic', 'Advanced', 'Satisfactory', 'Exceptional'), 'Proficient'),
    (('Basic', 'Advanced', 'Above-Average', 'Limited'), 'Competent'),
    (('Basic', 'Advanced', 'Above-Average', 'Basic'), 'Proficient'),
    (('Basic', 'Advanced', 'Above-Average', 'Proactive'), 'Proficient'),
    (('Basic', 'Advanced', 'Above-Average', 'Exceptional'), 'Proficient'),
    (('Basic', 'Advanced', 'High', 'Limited'), 'Competent'),
    (('Basic', 'Advanced', 'High', 'Basic'), 'Proficient'),
    (('Basic', 'Advanced', 'High', 'Proactive'), 'Proficient'),
    (('Basic', 'Advanced', 'High', 'Exceptional'), 'Proficient'),
    (('Basic', 'Exceptional', 'Low', 'Limited'), 'Novice'),
    (('Basic', 'Exceptional', 'Low', 'Basic'), 'Competent'),
    (('Basic', 'Exceptional', 'Low', 'Proactive'), 'Proficient'),
    (('Basic', 'Exceptional', 'Low', 'Exceptional'), 'Proficient'),
    (('Basic', 'Exceptional', 'Satisfactory', 'Limited'), 'Novice'),
    (('Basic', 'Exceptional', 'Satisfactory', 'Basic'), 'Competent'),
    (('Basic', 'Exceptional', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('Basic', 'Exceptional', 'Satisfactory', 'Exceptional'), 'Proficient'),
    (('Basic', 'Exceptional', 'Above-Average', 'Limited'), 'Competent'),
    (('Basic', 'Exceptional', 'Above-Average', 'Basic'), 'Proficient'),
    (('Basic', 'Exceptional', 'Above-Average', 'Proactive'), 'Proficient'),
    (('Basic', 'Exceptional', 'Above-Average', 'Exceptional'), 'Proficient'),
    (('Basic', 'Exceptional', 'High', 'Limited'), 'Proficient'),
    (('Basic', 'Exceptional', 'High', 'Basic'), 'Proficient'),
    (('Basic', 'Exceptional', 'High', 'Proactive'), 'Proficient'),
    (('Basic', 'Exceptional', 'High', 'Exceptional'), 'Proficient'),
    (('Moderate', 'Basic', 'Low', 'Limited'), 'Novice'),
    (('Moderate', 'Basic', 'Low', 'Basic'), 'Competent'),
    (('Moderate', 'Basic', 'Low', 'Proactive'), 'Proficient'),
    (('Moderate', 'Basic', 'Low', 'Exceptional'), 'Proficient'),
    (('Moderate', 'Basic', 'Satisfactory', 'Limited'), 'Novice'),
    (('Moderate', 'Basic', 'Satisfactory', 'Basic'), 'Competent'),
    (('Moderate', 'Basic', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('Moderate', 'Basic', 'Satisfactory', 'Exceptional'), 'Proficient'),
    (('Moderate', 'Basic', 'Above-Average', 'Limited'), 'Competent'),
    (('Moderate', 'Basic', 'Above-Average', 'Basic'), 'Proficient'),
    (('Moderate', 'Basic', 'Above-Average', 'Proactive'), 'Proficient'),
    (('Moderate', 'Basic', 'Above-Average', 'Exceptional'), 'Proficient'),
    (('Moderate', 'Basic', 'High', 'Limited'), 'Proficient'),
    (('Moderate', 'Basic', 'High', 'Basic'), 'Proficient'),
    (('Moderate', 'Basic', 'High', 'Proactive'), 'Proficient'),
    (('Moderate', 'Basic', 'High', 'Exceptional'), 'Proficient'),
    (('Moderate', 'Effective', 'Low', 'Limited'), 'Novice'),
    (('Moderate', 'Effective', 'Low', 'Basic'), 'Competent'),
    (('Moderate', 'Effective', 'Low', 'Proactive'), 'Proficient'),
    (('Moderate', 'Effective', 'Low', 'Exceptional'), 'Proficient'),
    (('Moderate', 'Effective', 'Satisfactory', 'Limited'), 'Competent'),
    (('Moderate', 'Effective', 'Satisfactory', 'Basic'), 'Proficient'),
    (('Moderate', 'Effective', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('Moderate', 'Effective', 'Satisfactory', 'Exceptional'), 'Proficient'),
    (('Moderate', 'Effective', 'Above-Average', 'Limited'), 'Competent'),
    (('Moderate', 'Effective', 'Above-Average', 'Basic'), 'Proficient'),
    (('Moderate', 'Effective', 'Above-Average', 'Proactive'), 'Proficient'),
    (('Moderate', 'Effective', 'Above-Average', 'Exceptional'), 'Proficient'),
    (('Moderate', 'Effective', 'High', 'Limited'), 'Proficient'),
    (('Moderate', 'Effective', 'High', 'Basic'), 'Proficient'),
    (('Moderate', 'Effective', 'High', 'Proactive'), 'Proficient'),
    (('Moderate', 'Effective', 'High', 'Exceptional'), 'Proficient'),
    (('Moderate', 'Advanced', 'Low', 'Limited'), 'Novice'),
    (('Moderate', 'Advanced', 'Low', 'Basic'), 'Competent'),
    (('Moderate', 'Advanced', 'Low', 'Proactive'), 'Proficient'),
    (('Moderate', 'Advanced', 'Low', 'Exceptional'), 'Proficient'),
    (('Moderate', 'Advanced', 'Satisfactory', 'Limited'), 'Competent'),
    (('Moderate', 'Advanced', 'Satisfactory', 'Basic'), 'Proficient'),
    (('Moderate', 'Advanced', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('Moderate', 'Advanced', 'Satisfactory', 'Exceptional'), 'Proficient'),
    (('Moderate', 'Advanced', 'Above-Average', 'Limited'), 'Proficient'),
    (('Moderate', 'Advanced', 'Above-Average', 'Basic'), 'Proficient'),
    (('Moderate', 'Advanced', 'Above-Average', 'Proactive'), 'Proficient'),
    (('Moderate', 'Advanced', 'Above-Average', 'Exceptional'), 'Proficient'),
    (('Moderate', 'Advanced', 'High', 'Limited'), 'Proficient'),
    (('Moderate', 'Advanced', 'High', 'Basic'), 'Proficient'),
    (('Moderate', 'Advanced', 'High', 'Proactive'), 'Proficient'),
    (('Moderate', 'Advanced', 'High', 'Exceptional'), 'Proficient'),
    (('Moderate', 'Exceptional', 'Low', 'Limited'), 'Competent'),
    (('Moderate', 'Exceptional', 'Low', 'Basic'), 'Proficient'),
    (('Moderate', 'Exceptional', 'Low', 'Proactive'), 'Proficient'),
    (('Moderate', 'Exceptional', 'Low', 'Exceptional'), 'Proficient'),
    (('Moderate', 'Exceptional', 'Satisfactory', 'Limited'), 'Competent'),
    (('Moderate', 'Exceptional', 'Satisfactory', 'Basic'), 'Proficient'),
    (('Moderate', 'Exceptional', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('Moderate', 'Exceptional', 'Satisfactory', 'Exceptional'), 'Proficient'),
    (('Moderate', 'Exceptional', 'Above-Average', 'Limited'), 'Proficient'),
    (('Moderate', 'Exceptional', 'Above-Average', 'Basic'), 'Proficient'),
    (('Moderate', 'Exceptional', 'Above-Average', 'Proactive'), 'Proficient'),
    (('Moderate', 'Exceptional', 'Above-Average', 'Exceptional'), 'Proficient'),
    (('Moderate', 'Exceptional', 'High', 'Limited'), 'Proficient'),
    (('Moderate', 'Exceptional', 'High', 'Basic'), 'Proficient'),
    (('Moderate', 'Exceptional', 'High', 'Proactive'), 'Proficient'),
    (('Moderate', 'Exceptional', 'High', 'Exceptional'), 'Advanced'),
    (('High', 'Basic', 'Low', 'Limited'), 'Novice'),
    (('High', 'Basic', 'Low', 'Basic'), 'Competent'),
    (('High', 'Basic', 'Low', 'Proactive'), 'Proficient'),
    (('High', 'Basic', 'Low', 'Exceptional'), 'Proficient'),
    (('High', 'Basic', 'Satisfactory', 'Limited'), 'Competent'),
    (('High', 'Basic', 'Satisfactory', 'Basic'), 'Proficient'),
    (('High', 'Basic', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('High', 'Basic', 'Satisfactory', 'Exceptional'), 'Proficient'),
    (('High', 'Basic', 'Above-Average', 'Limited'), 'Proficient'),
    (('High', 'Basic', 'Above-Average', 'Basic'), 'Proficient'),
    (('High', 'Basic', 'Above-Average', 'Proactive'), 'Proficient'),
    (('High', 'Basic', 'Above-Average', 'Exceptional'), 'Proficient'),
    (('High', 'Basic', 'High', 'Limited'), 'Proficient'),
    (('High', 'Basic', 'High', 'Basic'), 'Proficient'),
    (('High', 'Basic', 'High', 'Proactive'), 'Proficient'),
    (('High', 'Basic', 'High', 'Exceptional'), 'Proficient'),
    (('High', 'Effective', 'Low', 'Limited'), 'Competent'),
    (('High', 'Effective', 'Low', 'Basic'), 'Proficient'),
    (('High', 'Effective', 'Low', 'Proactive'), 'Proficient'),
    (('High', 'Effective', 'Low', 'Exceptional'), 'Proficient'),
    (('High', 'Effective', 'Satisfactory', 'Limited'), 'Competent'),
    (('High', 'Effective', 'Satisfactory', 'Basic'), 'Proficient'),
    (('High', 'Effective', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('High', 'Effective', 'Satisfactory', 'Exceptional'), 'Proficient'),
    (('High', 'Effective', 'Above-Average', 'Limited'), 'Proficient'),
    (('High', 'Effective', 'Above-Average', 'Basic'), 'Proficient'),
    (('High', 'Effective', 'Above-Average', 'Proactive'), 'Proficient'),
    (('High', 'Effective', 'Above-Average', 'Exceptional'), 'Proficient'),
    (('High', 'Effective', 'High', 'Limited'), 'Proficient'),
    (('High', 'Effective', 'High', 'Basic'), 'Proficient'),
    (('High', 'Effective', 'High', 'Proactive'), 'Proficient'),
    (('High', 'Effective', 'High', 'Exceptional'), 'Advanced'),
    (('High', 'Advanced', 'Low', 'Limited'), 'Competent'),
    (('High', 'Advanced', 'Low', 'Basic'), 'Proficient'),
    (('High', 'Advanced', 'Low', 'Proactive'), 'Proficient'),
    (('High', 'Advanced', 'Low', 'Exceptional'), 'Proficient'),
    (('High', 'Advanced', 'Satisfactory', 'Limited'), 'Proficient'),
    (('High', 'Advanced', 'Satisfactory', 'Basic'), 'Proficient'),
    (('High', 'Advanced', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('High', 'Advanced', 'Satisfactory', 'Exceptional'), 'Proficient'),
    (('High', 'Advanced', 'Above-Average', 'Limited'), 'Proficient'),
    (('High', 'Advanced', 'Above-Average', 'Basic'), 'Proficient'),
    (('High', 'Advanced', 'Above-Average', 'Proactive'), 'Proficient'),
    (('High', 'Advanced', 'Above-Average', 'Exceptional'), 'Proficient'),
    (('High', 'Advanced', 'High', 'Limited'), 'Proficient'),
    (('High', 'Advanced', 'High', 'Basic'), 'Proficient'),
    (('High', 'Advanced', 'High', 'Proactive'), 'Proficient'),
    (('High', 'Advanced', 'High', 'Exceptional'), 'Advanced'),
    (('High', 'Exceptional', 'Low', 'Limited'), 'Competent'),
    (('High', 'Exceptional', 'Low', 'Basic'), 'Proficient'),
    (('High', 'Exceptional', 'Low', 'Proactive'), 'Proficient'),
    (('High', 'Exceptional', 'Low', 'Exceptional'), 'Proficient'),
    (('High', 'Exceptional', 'Satisfactory', 'Limited'), 'Proficient'),
    (('High', 'Exceptional', 'Satisfactory', 'Basic'), 'Proficient'),
    (('High', 'Exceptional', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('High', 'Exceptional', 'Satisfactory', 'Exceptional'), 'Proficient'),
    (('High', 'Exceptional', 'Above-Average', 'Limited'), 'Proficient'),
    (('High', 'Exceptional', 'Above-Average', 'Basic'), 'Proficient'),
    (('High', 'Exceptional', 'Above-Average', 'Proactive'), 'Proficient'),
    (('High', 'Exceptional', 'Above-Average', 'Exceptional'), 'Advanced'),
    (('High', 'Exceptional', 'High', 'Limited'), 'Proficient'),
    (('High', 'Exceptional', 'High', 'Basic'), 'Proficient'),
    (('High', 'Exceptional', 'High', 'Proactive'), 'Proficient'),
    (('High', 'Exceptional', 'High', 'Exceptional'), 'Advanced'),
    (('Very-High', 'Basic', 'Low', 'Limited'), 'Competent'),
    (('Very-High', 'Basic', 'Low', 'Basic'), 'Proficient'),
    (('Very-High', 'Basic', 'Low', 'Proactive'), 'Proficient'),
    (('Very-High', 'Basic', 'Low', 'Exceptional'), 'Proficient'),
    (('Very-High', 'Basic', 'Satisfactory', 'Limited'), 'Proficient'),
    (('Very-High', 'Basic', 'Satisfactory', 'Basic'), 'Proficient'),
    (('Very-High', 'Basic', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('Very-High', 'Basic', 'Satisfactory', 'Exceptional'), 'Proficient'),
    (('Very-High', 'Basic', 'Above-Average', 'Limited'), 'Proficient'),
    (('Very-High', 'Basic', 'Above-Average', 'Basic'), 'Proficient'),
    (('Very-High', 'Basic', 'Above-Average', 'Proactive'), 'Proficient'),
    (('Very-High', 'Basic', 'Above-Average', 'Exceptional'), 'Advanced'),
    (('Very-High', 'Basic', 'High', 'Limited'), 'Proficient'),
    (('Very-High', 'Basic', 'High', 'Basic'), 'Proficient'),
    (('Very-High', 'Basic', 'High', 'Proactive'), 'Proficient'),
    (('Very-High', 'Basic', 'High', 'Exceptional'), 'Advanced'),
    (('Very-High', 'Effective', 'Low', 'Limited'), 'Proficient'),
    (('Very-High', 'Effective', 'Low', 'Basic'), 'Proficient'),
    (('Very-High', 'Effective', 'Low', 'Proactive'), 'Proficient'),
    (('Very-High', 'Effective', 'Low', 'Exceptional'), 'Proficient'),
    (('Very-High', 'Effective', 'Satisfactory', 'Limited'), 'Proficient'),
    (('Very-High', 'Effective', 'Satisfactory', 'Basic'), 'Proficient'),
    (('Very-High', 'Effective', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('Very-High', 'Effective', 'Satisfactory', 'Exceptional'), 'Proficient'),
    (('Very-High', 'Effective', 'Above-Average', 'Limited'), 'Proficient'),
    (('Very-High', 'Effective', 'Above-Average', 'Basic'), 'Proficient'),
    (('Very-High', 'Effective', 'Above-Average', 'Proactive'), 'Proficient'),
    (('Very-High', 'Effective', 'Above-Average', 'Exceptional'), 'Advanced'),
    (('Very-High', 'Effective', 'High', 'Limited'), 'Proficient'),
    (('Very-High', 'Effective', 'High', 'Basic'), 'Proficient'),
    (('Very-High', 'Effective', 'High', 'Proactive'), 'Advanced'),
    (('Very-High', 'Effective', 'High', 'Exceptional'), 'Advanced'),
    (('Very-High', 'Advanced', 'Low', 'Limited'), 'Proficient'),
    (('Very-High', 'Advanced', 'Low', 'Basic'), 'Proficient'),
    (('Very-High', 'Advanced', 'Low', 'Proactive'), 'Proficient'),
    (('Very-High', 'Advanced', 'Low', 'Exceptional'), 'Proficient'),
    (('Very-High', 'Advanced', 'Satisfactory', 'Limited'), 'Proficient'),
    (('Very-High', 'Advanced', 'Satisfactory', 'Basic'), 'Proficient'),
    (('Very-High', 'Advanced', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('Very-High', 'Advanced', 'Satisfactory', 'Exceptional'), 'Advanced'),
    (('Very-High', 'Advanced', 'Above-Average', 'Limited'), 'Proficient'),
    (('Very-High', 'Advanced', 'Above-Average', 'Basic'), 'Proficient'),
    (('Very-High', 'Advanced', 'Above-Average', 'Proactive'), 'Proficient'),
    (('Very-High', 'Advanced', 'Above-Average', 'Exceptional'), 'Advanced'),
    (('Very-High', 'Advanced', 'High', 'Limited'), 'Proficient'),
    (('Very-High', 'Advanced', 'High', 'Basic'), 'Proficient'),
    (('Very-High', 'Advanced', 'High', 'Proactive'), 'Advanced'),
    (('Very-High', 'Advanced', 'High', 'Exceptional'), 'Exceptional'),
    (('Very-High', 'Exceptional', 'Low', 'Limited'), 'Proficient'),
    (('Very-High', 'Exceptional', 'Low', 'Basic'), 'Proficient'),
    (('Very-High', 'Exceptional', 'Low', 'Proactive'), 'Proficient'),
    (('Very-High', 'Exceptional', 'Low', 'Exceptional'), 'Proficient'),
    (('Very-High', 'Exceptional', 'Satisfactory', 'Limited'), 'Proficient'),
    (('Very-High', 'Exceptional', 'Satisfactory', 'Basic'), 'Proficient'),
    (('Very-High', 'Exceptional', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('Very-High', 'Exceptional', 'Satisfactory', 'Exceptional'), 'Advanced'),
    (('Very-High', 'Exceptional', 'Above-Average', 'Limited'), 'Proficient'),
    (('Very-High', 'Exceptional', 'Above-Average', 'Basic'), 'Proficient'),
    (('Very-High', 'Exceptional', 'Above-Average', 'Proactive'), 'Advanced'),
    (('Very-High', 'Exceptional', 'Above-Average', 'Exceptional'), 'Advanced'),
    (('Very-High', 'Exceptional', 'High', 'Limited'), 'Proficient'),
    (('Very-High', 'Exceptional', 'High', 'Basic'), 'Proficient'),
    (('Very-High', 'Exceptional', 'High', 'Proactive'), 'Advanced'),
    (('Very-High', 'Exceptional', 'High', 'Exceptional'), 'Exceptional'),
    (('Exceptional', 'Basic', 'Low', 'Limited'), 'Proficient'),
    (('Exceptional', 'Basic', 'Low', 'Basic'), 'Proficient'),
    (('Exceptional', 'Basic', 'Low', 'Proactive'), 'Proficient'),
    (('Exceptional', 'Basic', 'Low', 'Exceptional'), 'Proficient'),
    (('Exceptional', 'Basic', 'Satisfactory', 'Limited'), 'Proficient'),
    (('Exceptional', 'Basic', 'Satisfactory', 'Basic'), 'Proficient'),
    (('Exceptional', 'Basic', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('Exceptional', 'Basic', 'Satisfactory', 'Exceptional'), 'Advanced'),
    (('Exceptional', 'Basic', 'Above-Average', 'Limited'), 'Proficient'),
    (('Exceptional', 'Basic', 'Above-Average', 'Basic'), 'Proficient'),
    (('Exceptional', 'Basic', 'Above-Average', 'Proactive'), 'Proficient'),
    (('Exceptional', 'Basic', 'Above-Average', 'Exceptional'), 'Advanced'),
    (('Exceptional', 'Basic', 'High', 'Limited'), 'Proficient'),
    (('Exceptional', 'Basic', 'High', 'Basic'), 'Proficient'),
    (('Exceptional', 'Basic', 'High', 'Proactive'), 'Advanced'),
    (('Exceptional', 'Basic', 'High', 'Exceptional'), 'Exceptional'),
    (('Exceptional', 'Effective', 'Low', 'Limited'), 'Proficient'),
    (('Exceptional', 'Effective', 'Low', 'Basic'), 'Proficient'),
    (('Exceptional', 'Effective', 'Low', 'Proactive'), 'Proficient'),
    (('Exceptional', 'Effective', 'Low', 'Exceptional'), 'Proficient'),
    (('Exceptional', 'Effective', 'Satisfactory', 'Limited'), 'Proficient'),
    (('Exceptional', 'Effective', 'Satisfactory', 'Basic'), 'Proficient'),
    (('Exceptional', 'Effective', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('Exceptional', 'Effective', 'Satisfactory', 'Exceptional'), 'Advanced'),
    (('Exceptional', 'Effective', 'Above-Average', 'Limited'), 'Proficient'),
    (('Exceptional', 'Effective', 'Above-Average', 'Basic'), 'Proficient'),
    (('Exceptional', 'Effective', 'Above-Average', 'Proactive'), 'Advanced'),
    (('Exceptional', 'Effective', 'Above-Average', 'Exceptional'), 'Advanced'),
    (('Exceptional', 'Effective', 'High', 'Limited'), 'Proficient'),
    (('Exceptional', 'Effective', 'High', 'Basic'), 'Proficient'),
    (('Exceptional', 'Effective', 'High', 'Proactive'), 'Advanced'),
    (('Exceptional', 'Effective', 'High', 'Exceptional'), 'Exceptional'),
    (('Exceptional', 'Advanced', 'Low', 'Limited'), 'Proficient'),
    (('Exceptional', 'Advanced', 'Low', 'Basic'), 'Proficient'),
    (('Exceptional', 'Advanced', 'Low', 'Proactive'), 'Proficient'),
    (('Exceptional', 'Advanced', 'Low', 'Exceptional'), 'Advanced'),
    (('Exceptional', 'Advanced', 'Satisfactory', 'Limited'), 'Proficient'),
    (('Exceptional', 'Advanced', 'Satisfactory', 'Basic'), 'Proficient'),
    (('Exceptional', 'Advanced', 'Satisfactory', 'Proactive'), 'Proficient'),
    (('Exceptional', 'Advanced', 'Satisfactory', 'Exceptional'), 'Advanced'),
    (('Exceptional', 'Advanced', 'Above-Average', 'Limited'), 'Proficient'),
    (('Exceptional', 'Advanced', 'Above-Average', 'Basic'), 'Proficient'),
    (('Exceptional', 'Advanced', 'Above-Average', 'Proactive'), 'Advanced'),
    (('Exceptional', 'Advanced', 'Above-Average', 'Exceptional'), 'Exceptional'),
    (('Exceptional', 'Advanced', 'High', 'Limited'), 'Proficient'),
    (('Exceptional', 'Advanced', 'High', 'Basic'), 'Advanced'),
    (('Exceptional', 'Advanced', 'High', 'Proactive'), 'Advanced'),
    (('Exceptional', 'Advanced', 'High', 'Exceptional'), 'Distinguished'),
    (('Exceptional', 'Exceptional', 'Low', 'Limited'), 'Proficient'),
    (('Exceptional', 'Exceptional', 'Low', 'Basic'), 'Proficient'),
    (('Exceptional', 'Exceptional', 'Low', 'Proactive'), 'Proficient'),
    (('Exceptional', 'Exceptional', 'Low', 'Exceptional'), 'Advanced'),
    (('Exceptional', 'Exceptional', 'Satisfactory', 'Limited'), 'Proficient'),
    (('Exceptional', 'Exceptional', 'Satisfactory', 'Basic'), 'Proficient'),
    (('Exceptional', 'Exceptional', 'Satisfactory', 'Proactive'), 'Advanced'),
    (('Exceptional', 'Exceptional', 'Satisfactory', 'Exceptional'), 'Advanced'),
    (('Exceptional', 'Exceptional', 'Above-Average', 'Limited'), 'Proficient'),
    (('Exceptional', 'Exceptional', 'Above-Average', 'Basic'), 'Proficient'),
    (('Exceptional', 'Exceptional', 'Above-Average', 'Proactive'), 'Advanced'),
    (('Exceptional', 'Exceptional', 'Above-Average', 'Exceptional'), 'Exceptional'),
    (('Exceptional', 'Exceptional', 'High', 'Limited'), 'Proficient'),
    (('Exceptional', 'Exceptional', 'High', 'Basic'), 'Advanced'),
    (('Exceptional', 'Exceptional', 'High', 'Proactive'), 'Exceptional'),
    (('Exceptional', 'Exceptional', 'High', 'Exceptional'), 'Distinguished'),
    
]
