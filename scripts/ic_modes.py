t2i_fc = (
    "<ins><b>Relighting with Foreground Condition</b></ins><br>"
    "Given a foreground image, generate a new background via <code>txt2img</code>"
    ", then blend them together while keeping the lighting conditions coherent."
)

t2i_fbc = (
    "<ins><b>Relighting with Foreground and Background Condition</b></ins><br>"
    "Extract the subject from the foreground image, then place it onto the background image"
    ", while keeping the lighting conditions coherent.<br>"
    "<code>Sampler</code> and <code>Steps</code> are important; while <code>Prompts</code> doesn't matter"
)

i2i_fc = (
    "<ins><b>Relighting with Conditions</b></ins><br>"
    "Given an input image, generate a new background with a conditioned lighting"
)
