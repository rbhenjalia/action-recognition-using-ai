keras.layers.GRU(
		units,
		activation='tanh',
		recurrent_activation='hard_sigmoid',
		use_bias=True,
		kernel_initializer='glorot_uniform',
		recurrent_initializer='orthogonal',
		bias_initializer='zeros',
		kernel_regularizer=None,
		recurrent_regularizer=None,
		bias_regularizer=None,
		activity_regularizer=None,
		kernel_constraint=None,
		recurrent_constraint=None,
		bias_constraint=None,
		dropout=0.0,
		recurrent_dropout=0.0,
		implementation=1,
		return_sequences=False,
		return_state=False,
		go_backwards=False,
		stateful=False,
		unroll=False,
		reset_after=False
	)


keras.layers.LSTM(
		units,
		activation='tanh', # default
		recurrent_activation='hard_sigmoid', # default
		use_bias=True, # default
		kernel_initializer='glorot_uniform', # 'glorot_normal' -- with seed
		recurrent_initializer='orthogonal',
		bias_initializer='zeros', # 'RandomNormal' -- with seed
		unit_forget_bias=True,
		kernel_regularizer=None,
		recurrent_regularizer=None,
		bias_regularizer=None,
		activity_regularizer=None,
		kernel_constraint=None,
		recurrent_constraint=None,
		bias_constraint=None,
		dropout=0.0, # 0.35
		recurrent_dropout=0.0, # 0.2
		implementation=1, # experiment
		return_sequences=False, # experiment
		return_state=False,
		go_backwards=False,
		stateful=False, # experiment
		unroll=False
	)





keras.layers.recurrent.LSTM(
	2048,
	kernel_initializer='glorot_uniform', # 'glorot_normal' -- with seed
	bias_initializer='zeros', # 'RandomNormal' -- with seed
	recurrent_dropout=0.0, # 0.2
	implementation=1, # experiment
	return_sequences=False, # experiment
	unit_forget_bias=True,
	return_sequences=False,
	input_shape=(1,2048),
	dropout=0.2
	)


