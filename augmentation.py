from imgaug import augmenters as iaa

affine_seq = iaa.Sequential([
    # General
    iaa.SomeOf((1, 2),
               [iaa.Fliplr(0.5), # horizontally flip 50% of all images
                iaa.Flipud(0.5), # vertically flip 50% of all images
                iaa.Affine(rotate=(0, 360), # rotate by 0 to 360 degrees
                           translate_percent=(-0.1, 0.1)), # translate by -0.1 to +0.1 percent (per axis)
                iaa.CropAndPad(percent=(-0.25, 0.25), pad_cval=0) # crop images by -25% to 25% of their height/width
                ]),
    # Deformations
    iaa.PiecewiseAffine(scale=(0.00, 0.06)) # move parts of the image around
], random_order=True)

color_seq = iaa.Sequential([
    # Color
    iaa.OneOf([
        iaa.Sequential([
            iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
            iaa.WithChannels(0, iaa.Add((0, 100))),
            iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
        iaa.Sequential([
            iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
            iaa.WithChannels(1, iaa.Add((0, 100))),
            iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
        iaa.Sequential([
            iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
            iaa.WithChannels(2, iaa.Add((0, 100))),
            iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
        iaa.WithChannels(0, iaa.Add((0, 100))),
        iaa.WithChannels(1, iaa.Add((0, 100))),
        iaa.WithChannels(2, iaa.Add((0, 100)))
    ])
], random_order=True)