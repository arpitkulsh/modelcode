# modelcode

model.eval()
model.config.return_dict = False   # makes sure we get a plain tuple back

torch.onnx.export(
    model,
    # either as a tuple of POSITIONAL args:
    args=(
      inputs["pixel_values"],
      inputs["input_ids"],
      inputs["attention_mask"],
      False                              # <-- return_loss positional
    ),
    # …or you can use kwargs to be explicit:
    # args=(),  
    # kwargs={
    #   "pixel_values": inputs["pixel_values"],
    #   "input_ids":    inputs["input_ids"],
    #   "attention_mask": inputs["attention_mask"],
    #   "return_loss": False
    # },
    f="siglip_combined.onnx",
    input_names=["pixel_values", "input_ids", "attention_mask", "return_loss"],
    output_names=["image_embeds", "text_embeds"],
    dynamic_axes={
      "pixel_values":   {0: "batch_size"},
      "input_ids":      {0: "batch_size"},
      "attention_mask": {0: "batch_size"},
      "image_embeds":   {0: "batch_size"},
      "text_embeds":    {0: "batch_size"},
    },
    opset_version=14,
    do_constant_folding=True
)
