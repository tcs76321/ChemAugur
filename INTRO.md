# How ChemAugur Works: From Atoms to Predictions

This document explains the chemistry and machine learning behind ChemAugur. It assumes you know how to code and remember some high school chemistry, but are rusty on concepts like dihedral angles, equivariance, and why geometry is all you need.

## 1. What Is a Molecule, Really?

A molecule is a collection of atoms held together by shared electrons. Each atom has a **type** (carbon, oxygen, hydrogen, etc.) identified by its atomic number `z`. And each atom has a **position** in 3D space `(x, y, z)`.

That's it. Those two pieces of information — `z` and `pos` — describe everything about a molecule. Everything else a chemist might talk about (bond types, angles, strain, hydrogen bonds) is a *consequence* of which atoms are present and where they are.

## 2. What Can You Tell From Positions?

### Distances Tell You Bond Types

Two atoms at a certain distance apart tells you what kind of bond they share:

| Distance | What it means |
|---|---|
| 1.54 Å | Carbon-carbon single bond |
| 1.34 Å | Carbon-carbon double bond |
| 1.20 Å | Carbon-carbon triple bond |
| 1.40 Å | Aromatic (like benzene — in between single and double) |
| 2.5–3.0 Å | Hydrogen bond (O–H···O) |
| 3.5–4.0 Å | π-stacking (parallel aromatic rings) |
| 0.96 Å | O–H covalent bond |

The model doesn't need to be told "this is a double bond." It learns that atoms at 1.34 Å behave differently from atoms at 1.54 Å.

### Angles Tell You Hybridization and Shape

Three atoms form an angle. The angle between bonds around a central atom tells you its **hybridization** — how its electron orbitals are arranged in space:

| Angle | Hybridization | Shape | Example |
|---|---|---|---|
| 109.5° | sp³ | Tetrahedral | Methane (CH₄), ethanol |
| 120° | sp² | Trigonal planar | Benzene, ethylene |
| 180° | sp | Linear | Acetylene, CO₂ |

The model learns that 120° angles around carbon correlate with aromatic systems, while 109.5° angles correlate with saturated (alkane-like) systems. It discovers this from the geometry data without being told.

Deviations from these ideal angles mean something too. Cyclopropane has 60° bond angles — far from the ideal 109.5°. That's **ring strain**, and it makes cyclopropane more reactive. The model can learn that compressed angles correlate with higher energy and different physical properties.

### Dihedrals Tell You 3D Shape and Flexibility

A **dihedral angle** involves four atoms in sequence (A–B–C–D). Imagine looking down the B–C bond — the dihedral is the angle between the A–B bond and the C–D bond when viewed from that direction.

```
        A
         \
          B
         / \
            C
           / \
              D
```

Think of it this way: if distances tell you *how far apart* atoms are, and angles tell you *what direction* the next bond goes, dihedrals tell you *how the molecule twists* in 3D.

Dihedrals matter because:

- **Conformational flexibility**: Butane (C₄H₁₀) can rotate around its central C–C bond. The *gauche* conformation (dihedral ~60°) has different properties than the *anti* conformation (dihedral ~180°). These different shapes have different energies and thus different boiling points, solubilities, etc.

- **Ring puckering**: Cyclohexane isn't flat — it puckers into a "chair" shape. The dihedral angles describe exactly how it puckers. This affects how the molecule packs in a liquid, which affects its bulk properties.

- **Stereochemistry**: Two molecules can have the same atoms connected in the same order but be mirror images (like left and right hands). Dihedral angles distinguish these — they're how the model "sees" chirality.

- **Torsional strain**: When bonds are twisted away from their preferred angles, the molecule stores energy. Dihedrals capture this.

### Many-Body Effects: Beyond Triplets and Quartets

Real chemistry isn't just pairs (distances) and triplets (angles) and quartets (dihedrals). Sometimes five or six atoms interact collectively:

- **Cooperative hydrogen bonding**: In water, hydrogen bonds strengthen each other. One water molecule hydrogen-bonding to a second makes the second a better hydrogen bond donor to a third. This is a 5+ atom effect.

- **Solvent cages**: A solute molecule surrounded by many solvent molecules in a specific arrangement. The collective geometry matters more than any single pairwise interaction.

MACE captures these through its **many-body expansion** — it systematically includes pair interactions, triplet interactions, quartet interactions, and optionally higher-order terms.

## 3. Why Not Just Use Bond Types and Labels?

Many chemistry ML projects encode features that chemists *inferred* from geometry: "this carbon is sp² hybridized," "this bond is aromatic," "this atom has formal charge +1." This approach has problems:

1. **It tells the model the answer.** If you tell the model a bond is aromatic, you've already decided how that bond behaves. The model can't discover that a particular geometric arrangement behaves *like* aromaticity without fitting the label — even if the chemistry is more subtle.

2. **Labels fail at the edges.** Transition metal complexes, 3-center-2-electron bonds (diborane), frustrated Lewis pairs, hypervalent iodine — these don't fit neatly into sp²/sp³ categories. The geometry is always correct, even when the labels are wrong.

3. **Labels are human constructs.** "Aromaticity" is a name we give to a geometric + electronic pattern. The pattern exists whether we name it or not. The model should learn the pattern, not the name.

## 4. How Does the Model Actually Work?

### Step 1: Generate 3D Geometry

Given a SMILES string like `"CCO"` (ethanol), we:
1. Parse it into a molecular graph (atoms and bonds) using RDKit
2. Generate a 3D shape using ETKDG — an algorithm that uses experimental data from the Cambridge Structural Database to predict likely 3D arrangements
3. Refine with MMFF94 — a physics-based force field that adjusts atom positions to minimize energy

The result is a physically reasonable 3D structure.

### Step 2: Extract Just `z` and `pos`

From the 3D molecule, we extract:
- `z`: a list of atomic numbers `[6, 6, 8, 1, 1, 1, 1, 1, 1]` (2 carbons, 1 oxygen, 6 hydrogens)
- `pos`: a 9×3 matrix of (x, y, z) coordinates in Angstroms

That's the entire input. No bond types, no angles, no dihedrals — those are computed internally by the model from `pos`.

### Step 3: MACE Processes the Geometry

MACE (Multi-body Atomic Cluster Expansion) is an equivariant graph neural network. Here's what it does internally:

1. **Build a radius graph**: Connect every atom to every other atom within a cutoff distance (default 12 Å). Atoms farther apart are ignored.

2. **Expand distances into features**: For each pair of connected atoms, the distance is expanded into a set of radial basis functions (RBFs) — smooth mathematical functions that represent the distance in a way the network can learn from. Think of it like representing a number in multiple different number bases simultaneously.

3. **Compute angles and dihedrals**: For each triplet of connected atoms, compute the angle. For each quartet, compute the dihedral. These are encoded using spherical harmonics — mathematical functions on a sphere that describe angular patterns.

4. **Equivariant message passing**: Information flows between atoms through the radius graph. Each atom's representation is updated based on its neighbors, the distances to those neighbors, and the angles between them. Crucially, the updates are **equivariant** — if you rotate the entire molecule, the atom representations rotate with it, and the final prediction (a scalar like boiling point) stays exactly the same.

5. **Many-body expansion**: The network systematically builds up representations of pairs, triplets, quartets, and optionally higher-order groups. This is what "many-body" means — it doesn't just look at atoms one at a time or in pairs, but in increasingly larger groups.

6. **Global pooling**: All atom representations are averaged together to produce a single vector representing the entire system.

7. **Output head**: A small neural network maps this vector to a single number — the predicted property (boiling point, solubility, etc.).

### What Does "Equivariant" Mean?

This is worth understanding. "Equivariant" means: if you rotate or translate the molecule, the model's internal representations transform in a predictable way. The final prediction (a scalar) is **invariant** — it doesn't change at all when you rotate the molecule.

This matters because a molecule's boiling point doesn't depend on which direction it's facing. A model that *wasn't* equivariant would have to learn this from data, wasting capacity. An equivariant model has this property built into its mathematics.

## 5. How Do Mixtures Work?

A mixture — like 50% ethanol and 50% water — is treated exactly the same as a pure substance. We just place more molecules in the same coordinate space:

1. Generate a 3D conformer for ethanol
2. Generate a 3D conformer for water
3. Place 5 ethanol molecules and 5 water molecules in a shared 3D box:
   - Each molecule gets a random position and random rotation
   - If any atoms are too close together (steric clash), reject that placement and try again
   - Optionally, run a quick force-field relaxation to let molecules settle into favorable positions
4. Combine all atomic numbers and positions into one big `z` and `pos`
5. Feed through MACE — the model sees 5×9 + 5×3 = 60 atoms, with both intra-molecular and inter-molecular distances, angles, and dihedrals

The model has **no concept of molecule identity**. It doesn't know which atoms belong to ethanol and which to water. It just sees atoms in space and learns that certain patterns of distances, angles, and dihedrals between certain elements correlate with certain properties. An O–H···O hydrogen bond between two ethanol molecules looks geometrically similar to an O–H···O hydrogen bond between ethanol and water. The model learns the pattern, not the labels.

This is why the same model works for both pure substances and mixtures — "mixture" is just a human category for "multiple species in the same space." The physics doesn't care.

## 6. What Properties Can We Predict?

Any physical property that depends on how molecules interact with each other:

- Boiling point
- Melting point
- Vapor pressure
- Solubility
- Partition coefficient (logP)
- Density
- Viscosity
- Activity coefficients (how non-ideal a mixture is)
- Azeotrope composition (mixtures that boil at constant composition)

These are all **bulk thermodynamic properties** — properties of large collections of molecules, not single molecules in isolation. They depend on how molecules pack together, form and break hydrogen bonds, and interact through van der Waals forces. All of these are geometric.

## 7. What Are the Limits?

**Conformer accuracy**: Our 3D geometries come from ETKDG + MMFF, which are fast approximations. They're physically reasonable but not as accurate as quantum chemistry (DFT). For some properties, this may matter.

**Single conformation**: We use one low-energy 3D shape per molecule. Real molecules at room temperature are vibrating and rotating through many shapes (conformations). Properties like entropy depend on how many shapes are accessible. Future work will sample multiple conformations.

**Cutoff radius**: We ignore interactions beyond 12-15 Å. This captures covalent bonds, hydrogen bonds, π-stacking, and the first few solvation shells. It misses long-range electrostatics, which matter for charged species. Future work will add Ewald summation or similar long-range corrections.

**No quantum effects**: MACE learns from geometry patterns but doesn't explicitly model electrons. Properties that depend on electronic excited states or radical chemistry are out of scope.

**Data requirements**: We need experimental data (measured boiling points, etc.) to train on. For novel molecules or mixtures, we can only predict what's similar to the training data. The model can interpolate but not extrapolate to completely new chemistry.

## 8. Why MACE Specifically?

We evaluated several architectures:

| Model | Distances | Angles | Dihedrals | Tensor Features | Many-Body |
|---|---|---|---|---|---|
| DimeNet++ | ✓ | ✓ | ✗ | ✗ | ✗ |
| GemNet | ✓ | ✓ | ✓ | ✗ | ✗ |
| EquiformerV2 | ✓ | ✓ | ✗ | ✓ | ✗ |
| **MACE** | **✓** | **✓** | **✓** | **✓** | **✓** |

Only MACE gives us all four geometric levels simultaneously: distances for bond types, angles for hybridization and strain, dihedrals for conformational flexibility and stereochemistry, tensor features for directional effects like polarity, and many-body terms for collective solvent effects.

It's also a mature, well-tested architecture (validated on quantum chemistry benchmarks), with an active open-source community. We wrap it in a clean `forward(z, pos, batch)` interface so we can swap to a different architecture later without changing the training pipeline.

## 9. Further Reading

- **MACE paper**: Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields" (NeurIPS 2022)
- **ACE theory**: Drautz, "Atomic cluster expansion for accurate and transferable interatomic potentials" (Physical Review B, 2019)
- **Equivariance**: Thomas et al., "Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds" (NeurIPS 2018)
- **Conformer generation**: Riniker & Landrum, "Better Informed Distance Geometry: Using What We Know To Improve Conformer Generation" (J. Chem. Inf. Model., 2015) — the ETKDG paper
- **Graph neural networks for chemistry**: Reiser et al., "Graph neural networks for materials science and chemistry" (Communications Materials, 2022) — a good survey

## 10. Glossary

| Term | Definition |
|---|---|
| **Atomic number (z)** | The number of protons in an atom's nucleus. Identifies the element. |
| **Angstrom (Å)** | 10⁻¹⁰ meters. Convenient unit for atomic distances. |
| **Conformer** | A specific 3D shape of a molecule. Most molecules can adopt many conformers by rotating around single bonds. |
| **Cutoff radius** | Maximum distance at which two atoms are considered "interacting." Beyond this, they're ignored to save computation. |
| **Dihedral angle** | The angle between two planes defined by four atoms in sequence. Describes twisting around a bond. |
| **Equivariance** | A property of a function: rotating the input rotates the output the same way. For scalar outputs, this becomes invariance — rotation doesn't change the result at all. |
| **ETKDG** | Experimental Torsion Knowledge Distance Geometry. An algorithm that generates 3D molecular shapes using experimental data about which torsion angles are common. |
| **Hydrogen bond** | An attractive interaction between a hydrogen atom bonded to an electronegative atom (like O or N) and another electronegative atom. Distance ~2.5-3.0 Å. |
| **MMFF94** | Merck Molecular Force Field. A physics-based method for calculating molecular energies and optimizing geometries. |
| **Many-body** | Considering interactions among groups of 3, 4, or more atoms simultaneously, rather than just pairwise. |
| **SMILES** | Simplified Molecular Input Line Entry System. A text string that encodes a molecular structure (e.g., `"CCO"` = ethanol). |
| **Spherical harmonics** | Mathematical functions on a sphere, analogous to sine/cosine waves on a circle. Used to encode angular information. |
| **Steric clash** | Two atoms too close together — their electron clouds overlap repulsively. Physically impossible at very short distances. |
| **Van der Waals forces** | Weak attractive forces between all atoms and molecules, caused by temporary fluctuations in electron distribution. |