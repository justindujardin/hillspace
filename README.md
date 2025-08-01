# Hill Space is All You Need

This repository contains all the source assets related to Hill Space is All You Need.

# Training

This repo contains a general training script that I used for research while writing the paper. It's not used directly in the experiments, but has some other features that might be interesting but didn't make it into the paper.

The default configuration trains all the **Arithmetic** and **Trigonometry** operators talked about in the paper. It uses hill space with snapping, and logs extensive metrics to Weights and Biases.

```python
python -m hillspace.train
```

## Training Features

Some interesting features that didn't make it are:

- There are MANY operators I didn't manage to "solve" for the paper, but haven't revisited with the final model formulation. They could be good starting points for new primitive exploration, etc. (see operator_specs.py)
- I struggled with trying to learn powers, but their optimal weights mostly exist in the "Valley of Sorrow" in Hill Space where they're tightly clustered together near gradient dead zones. There are other formulations of spaces I thought might accomodate those fractional optimal weights better. Perhaps now with all the remaining error accounted for it would be worth exploring them again.

# Experiments

We conducted 5 experiments in the paper to demonstrate Hill Space's unique properties and to back up the rather bold claims we make about precision and what error remains in our method.

**NOTE**: Experiments 3-5 all have the results from the paper precalculated and will only print them out when run. To reproduce the results yourself you need to go to the bottom of the desired experiment and uncomment the commented out line that calculates the result data for aggregation

## Experiment 1 - Direct Weight Enumeration

```python
python -m hillspace.experiments.experiment_neural_calc 1339.7364 - 2.7364
```

## Experiment 2 - Learning Divison in 60 Seconds

```python
python -m hillspace.experiments.experiment_train_division
```

## Experiment 3 - Comparison with iNALU

```python
python -m hillspace.experiments.experiment_inalu
```

## Experiment 4 - Error Analysis and Attribution

```python
python -m hillspace.experiments.experiment_error
```

## Experiment 5 - Weight Initialization Robustness

```python
python -m hillspace.experiments.experiment_init
```

# Visualizations

The visualizations in the paper comes from both Python scripts that output images, and a set of React widgets that export SVGs for each mathematical primitive.

## Hill Space Visualization

This figure maps Hill Space into a 3d image to help users understand the hill shape it produces, and how the inputs relate to locations in the space.

```bash
python -m hillspace.figures.hill_space_visualization
```

![Hill Space](./paper/images/hill_space.png)

## Tanh/Sigmoid Interaction Visualization

This figure demonstrates visually how the tanh and sigmoid functions interact to produce constrained weights. It highlights how tanh acts to select the signal, and sigmoid gates it up or down.

```bash
python -m hillspace.figures.activation_visualization
```

![Tanh Sigmoid Interactions](./paper/images/hill_space_cross_sections.svg)

## Mathematical Primitives

All four primitives introduced in the paper are visualized using React widgets that save PNG and SVG renderings for external use, and provide an interactive experience to allow a more intuitive way of digesting what's happening.

Open the desired file in your web browser to interact with the widgets, and click "Download PNG" or "Download SVG" if you want to capture a particular configuration.

### Additive Primitive

![Additive Primitive](./paper/images/additive_primitive.svg)

```bash
open web/additive_primitive.html
```

### Exponential Primitive

![Exponential Primitive](./paper/images/exponential_primitive.svg)

```bash
open web/exponential_primitive.html
```

### Unit Circle Primitive

![Unit Circle Primitive](./paper/images/rotation_transformation.svg)

```bash
open web/unit_circle_primitive.html
```

### Trigonometric Products Primitive

![Trigonometric Products Primitive](./paper/images/trigonometric_product_primitive.svg)

```bash
open web/trigonometric_products_primitive.html
```

# License

- Code: MIT License (see [LICENSE.md](LICENSE.md))
- Paper contents and documentation: CC-BY 4.0

## Citation

If you use Hill Space in your research, please cite:

```bibtex
@misc{dujardin2025hillspace,
  author = {DuJardin, Justin},
  title = {Hill Space is All You Need},
  year = {2025},
  month = {7},
  howpublished = {\url{https://github.com/justindujardin/hillspace}},
  note = {Preprint submitted to TechRxiv}
}
```

### Text citation:

Justin DuJardin. "Hill Space is All You Need". July 2025. GitHub repository: https://github.com/justindujardin/hillspace. Preprint submitted to TechRxiv.

_Note: If you're interested in sponsoring this work on arXiv, please reach out._
