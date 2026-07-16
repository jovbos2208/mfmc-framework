#!/bin/bash
#SBATCH --job-name=piclas_test
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --cpus-per-task=1
#SBATCH --output=piclas_slurm-%A.out
#SBATCH --error=piclas_slurm-%A.err
#SBATCH --mail-type=NONE
# Optional site settings:
# #SBATCH --account=<YOUR_SLURM_ACCOUNT>
# #SBATCH --mail-user=<YOUR_EMAIL>

# Anzahl der Kerne pro Node
nodecores=36

# Gesamtzahl der Kerne berechnen
ncores=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

# Wechsel in das Einreichungsverzeichnis:
echo "Arbeitsverzeichnis: $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

# Laden der benötigten Module:
module load gcc/12.3.0
module load openmpi/4.1.5
module load hdf5/1.12.2   # Bitte ggf. die korrekte HDF5-Version anpassen
module load openblas/0.3.23


# Ausführen des Programms 
#mpirun -np $ncores ./piclas parameter.ini DSMC1.ini
./piclas2vtk Cube_DSMCSurfState*
