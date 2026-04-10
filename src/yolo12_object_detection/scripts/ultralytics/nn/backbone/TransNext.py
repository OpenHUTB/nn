try:
    import swattention
    from ultralytics.nn.backbone.TransNeXt.TransNext_cuda import *
except ImportError as e:
    from ultralytics.nn.backbone.TransNeXt.TransNext_native import *
    pass

