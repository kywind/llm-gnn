from llm.llm import LLM
from config import gen_args

args = gen_args()
llm = LLM(args)
material_dict = {}
obj_list = ['a mouse', 'a keyboard', 'a pen', 'a box', 'a cup', 'a cup mat']
for obj_name in obj_list:
    material_prompt = " ".join([
        "Classify the objects in the image as rigid objects, granular objects, deformable objects, or rope.",
        "Respond unknown if you are not sure."
        "\n\nQuery: coffee bean. Answer: granular.",
        "\n\nQuery: rope. Answer: rope.",
        "\n\nQuery: a wooden block. Answer: rigid.",
        "\n\nQuery: a banana. Answer: rigid.",
        "\n\nQuery: an orange. Answer: rigid.",
        "\n\nQuery: play-doh. Answer: deformable.",
        "\n\nQuery: sand. Answer: granular.",
        "\n\nQuery: a bottle. Answer: rigid.",
        "\n\nQuery: a t-shirt. Answer: deformable.",
        "\n\nQuery: rice. Answer: granular.",
        "\n\nQuery: laptop. Answer: rigid.",
        "\n\nQuery: " + obj_name + ". Answer:",
    ])
    material = llm.query(material_prompt)
    material = material.rstrip('.')
    material_dict[obj_name] = material
print(material_dict)
