# Paper 1 sphere reference candidates

## Recommended primary references

1. G. S. Springer and S. W. Tsai, Effect of thermal accommodation on cylinder and sphere drag in free molecule flow, AIAA Journal 2(1), 126-128 (1964), DOI: 10.2514/3.2238. This is the most directly matched primary source because it treats sphere drag and thermal accommodation explicitly.
2. L. H. Sentman, Free Molecule Flow Theory and Its Application to the Determination of Aerodynamic Forces (1961), DOI: 10.21236/AD0265409. This is the foundational source for the panel law implemented by the Sentman branch.
3. D. L. Whitfield and W. B. Stephenson, Sphere Drag in the Free-Molecular and Transitional Flow Regimes (1970), DOI: 10.21236/AD0704122. Use as an independent sphere-drag cross-check and for the free-molecular-to-transition boundary.
4. M. I. Kussoy, D. A. Stewart, and C. C. Horstman, Sphere drag in near-free-molecule hypersonic flow, AIAA Journal 8(11), 2104-2105 (1970), DOI: 10.2514/3.6070. This is a near-free-molecular experimental/computational comparison, not the primary analytical limit.

## Current evidence status

Crossref metadata and NASA NTRS catalogue records were checked. The Springer-Tsai and AIAA articles are paywalled in the sources found; the full equation and its exact Cd normalization have therefore not yet been transcribed into a verification config. No numerical literature value is claimed yet.

## Required extraction before freezing a target

Obtain and inspect the full text of Springer and Tsai and Sentman. Record: speed-ratio definition; incident gas temperature; wall temperature; projected reference area; diffuse/specular mixture; thermal-accommodation definition; high-speed limiting value; and whether the reported coefficient includes incident and re-emitted molecular momentum. Reproduce the formula independently, then compare it with direct numerical integration of the repository Sentman panel law on the fine sphere.

The reference test is accepted only when the literature normalization matches Aref = pi D squared over four and the deterministic configuration state exactly. Until then, the 2 percent TPMC/Sentman and 3 percent DSMC thresholds in verification_matrix.yaml are provisional preregistration targets rather than pass/fail results.
