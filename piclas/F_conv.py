import numpy as np
import pyvista as pv 
import matplotlib.pyplot as plt

F_step = []
time = []
for i in range(1,41):
    data = pv.read(f'output{i}.vtu')

    F = np.array(data.cell_data['Total_ForcePerArea'])[:,1]
    F_step.append(np.mean(F[np.where(F>1e-3)]))
    time.append(i*2.5e-4)

plt.plot(time,F_step,'.')
plt.xlabel('Time[s]')
plt.ylabel('Force per Area [N/m²]')
plt.grid()
plt.savefig('F_conv.png')
