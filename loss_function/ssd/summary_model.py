from keras.utils import plot_model


def summary_and_png(model, summary=True, to_png=False, png_file=None):
    if summary:
        model.summary()
    if to_png:
        plot_model(model, to_file='summary/'+png_file, show_shapes=True)