def ae(input_shape,filename,dropout=0.2,learning_rate=1e-4):
    
    tf.keras.backend.clear_session()

    activation_func = 'relu'

    def conv1d_pool_block(filters,kernel,activation_func,x):       
        x = Conv1D(filters, kernel, activation=activation_func,padding='same')(x)
        x = MaxPooling1D((2), padding='same')(x)
        return x

    def conv1d_main_block(filters,kernel,activation_func,x):
        x = Conv1D(filters, kernel, activation=activation_func,padding='same')(x)
        return x

    def conv1d_upsa_block(filters,kernel,activation_func,x):
        x = Conv1D(filters, kernel, activation=activation_func,padding='same')(x)
        x = UpSampling1D(2)(x)
        return x

    #----------------- Input

    x_inputs = Input(shape=input_shape)
    x = x_inputs

    #----------------- Encoder

    x = conv1d_pool_block(512,8,activation_func,x)
    x = conv1d_main_block(512,7,activation_func,x)
    
    x = conv1d_pool_block(256,6,activation_func,x)
    x = conv1d_main_block(256,5,activation_func,x)

    x = conv1d_pool_block(128,4,activation_func,x)
    x = conv1d_main_block(128,3,activation_func,x)

    x = conv1d_main_block(64,2,activation_func,x)
    x = conv1d_main_block(64,2,activation_func,x)

    x = conv1d_upsa_block(128,3,activation_func,x)
    x = conv1d_main_block(128,4,activation_func,x)

    x = conv1d_upsa_block(256,5,activation_func,x)
    x = conv1d_main_block(256,6,activation_func,x)
    
    x = conv1d_upsa_block(512,7,activation_func,x)
    x = conv1d_main_block(512,8,activation_func,x)

    #----------------- Output
    x = Dropout(dropout)(x)
    x = Dense(1)(x)

    model = Model(inputs=x_inputs, outputs=x, name='AE')

    opt = Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy','mean_squared_error'])

    model_filepath = filename
    model.save(model_filepath,overwrite=True,include_optimizer=True)

    return model
