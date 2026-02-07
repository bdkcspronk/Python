import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GLU import *
import time
import math
import random
from collections import defaultdict, deque

# ==============================================================================
# Visualization
# ==============================================================================
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 1000
VOXEL_SIZE = 5.0

# ==============================================================================
# COLORS
# ==============================================================================
COLOR_BACKGROUND   = (.1, 0.1, 0.1, 1.0)  # RGBA
COLOR_SPIN_UP   = (0.8, 0.8, 0.8)  # light gray
COLOR_SPIN_DOWN = (0.3, 0.5, 0.5)  # dark gray


# ==============================================================================
# 3D ISING MODEL PARAMETERS
# ==============================================================================
LATTICE_H = 20
LATTICE_W = 20
LATTICE_L = 20

TEMPERATURE = 50.0  # Temperature in units of J/k_B
J_COUPLING = 2.0   # Coupling constant
EXTERNAL_FIELD = 0.0  # External magnetic field

DISPLAY_UPDATE_INTERVAL = 5000  # MC steps between statistics display

# ==============================================================================
# ISING MODEL CLASS
# ==============================================================================
class IsingModel3D:
    """3D Ising model with Metropolis algorithm"""
    
    def __init__(self, H, W, L, temperature, j_coupling=1.0, external_field=0.0):
        self.H = H
        self.W = W
        self.L = L
        self.temperature = temperature
        self.j_coupling = j_coupling
        self.external_field = external_field
        
        # Initialize lattice with random spins (-1 or +1)
        self.spins = np.random.choice([-1, 1], size=(H, W, L))
        
        # Statistics
        self.energy_history = []
        self.magnetization_history = []
        self.step_count = 0
        
    def get_neighbors_energy(self, i, j, k):
        """Calculate energy from neighbors for site (i,j,k)"""
        spin = self.spins[i, j, k]
        neighbors_sum = 0
        
        # Periodic boundary conditions
        for di, dj, dk in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            ni = (i + di) % self.H
            nj = (j + dj) % self.W
            nk = (k + dk) % self.L

            neighbors_sum += self.spins[ni, nj, nk]
        
        energy = -self.j_coupling * spin * neighbors_sum - self.external_field * spin
        return energy
    
    def calculate_total_energy(self):
        """Calculate total system energy"""
        energy = 0.0
        for i in range(self.H):
            for j in range(self.W):
                for k in range(self.L):
                    energy += self.get_neighbors_energy(i, j, k)

        return energy / 2.0  # Divide by 2 to avoid double counting
    
    def calculate_magnetization(self):
        """Calculate total magnetization (net spin)"""
        return np.sum(self.spins) / (self.H * self.W * self.L)
    
    def metropolis_step(self):
        """Perform one Metropolis MC step"""
        # Select random site
        i = random.randint(0, self.H - 1)
        j = random.randint(0, self.W - 1)
        k = random.randint(0, self.L - 1)
        
        # Calculate energy before flip
        energy_before = self.get_neighbors_energy(i, j, k)
        
        # Flip spin
        self.spins[i, j, k] *= -1
        
        # Calculate energy after flip
        energy_after = self.get_neighbors_energy(i, j, k)
        
        # Metropolis acceptance criterion
        dE = energy_after - energy_before
        if dE > 0:
            # Reject with probability based on Boltzmann factor
            if random.random() > np.exp(-dE / self.temperature):
                self.spins[i, j, k] *= -1  # Flip back
        
        self.step_count += 1
    
    def run_steps(self, num_steps):
        """Run multiple Monte Carlo steps"""
        for _ in range(num_steps):
            self.metropolis_step()
    
    def get_statistics(self):
        """Get current energy and magnetization"""
        energy = self.calculate_total_energy()
        magnetization = self.calculate_magnetization()
        
        self.energy_history.append(energy)
        self.magnetization_history.append(magnetization)
        
        return energy, magnetization

# ==============================================================================
# POLARIZATION ANALYSIS
# ==============================================================================
class PolarizationAnalyzer:
    """Analyze polarization regions and domains"""
    
    @staticmethod
    def get_polarization_magnitude(spins):
        """Calculate polarization magnitude (0 to 1)"""
        return np.abs(np.sum(spins)) / spins.size
    
    @staticmethod
    def get_polarization_vector(spins):
        """Get polarization vector components"""
        avg_spin = np.sum(spins) / spins.size
        # In 3D, we can analyze slice-by-slice polarization
        polarization = {
            'total': avg_spin,
            'magnitude': np.abs(avg_spin)
        }
        return polarization
    
    @staticmethod
    def find_domains(spins, threshold=0.5):
        """Find connected regions of same spin using flood fill
        Returns: list of domain information"""
        
        H, W, L = spins.shape
        visited = np.zeros((H, W, L), dtype=bool)
        domains = []
        
        def flood_fill(start_i, start_j, start_k, target_spin):
            """BFS to find connected component"""
            queue = deque([(start_i, start_j, start_k)])
            visited[start_i, start_j, start_k] = True
            domain_spins = []
            
            while queue:
                i, j, k = queue.popleft()
                domain_spins.append((i, j, k))
                
                # Check 6 neighbors (3D)
                for di, dj, dk in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                    ni = (i + di) % H
                    nj = (j + dj) % W
                    nk = (k + dk) % L
                    
                    if not visited[ni, nj, nk] and spins[ni, nj, nk] == target_spin:
                        visited[ni, nj, nk] = True
                        queue.append((ni, nj, nk))
            
            return domain_spins
        
        # Find all domains
        for i in range(H):
            for j in range(W):
                for k in range(L):
                    if not visited[i, j, k]:
                        target_spin = spins[i, j, k]
                        domain = flood_fill(i, j, k, target_spin)
                        domain_volume = len(domain)
                        domain_spin = target_spin
                        
                        domains.append({
                            'spin': domain_spin,
                            'volume': domain_volume,
                            'positions': domain,
                            'fraction': domain_volume / spins.size
                        })
        
        return domains
    
    @staticmethod
    def get_domain_statistics(domains):
        """Calculate statistics about domains"""
        if not domains:
            return {}
        
        volumes = [d['volume'] for d in domains]
        spin_up_domains = [d for d in domains if d['spin'] == 1]
        spin_down_domains = [d for d in domains if d['spin'] == -1]
        
        stats = {
            'num_domains': len(domains),
            'num_spin_up': len(spin_up_domains),
            'num_spin_down': len(spin_down_domains),
            'avg_domain_size': np.mean(volumes),
            'max_domain_size': np.max(volumes),
            'min_domain_size': np.min(volumes),
            'largest_domain_spin_up': max([d['volume'] for d in spin_up_domains]) if spin_up_domains else 0,
            'largest_domain_spin_down': max([d['volume'] for d in spin_down_domains]) if spin_down_domains else 0,
        }
        return stats


class IsingSpin3DVisualizer:
    """OpenGL visualization of 3D Ising model with instanced rendering"""

    def __init__(self, model, analyzer):
        self.model = model
        self.analyzer = analyzer
        self.voxel_size = VOXEL_SIZE

        self.lattice_radius = 0.5 * math.sqrt(
            (self.model.H * self.voxel_size) ** 2 +
            (self.model.W * self.voxel_size) ** 2 +
            (self.model.L * self.voxel_size) ** 2
        )

        self.surface_indices = []
        for i in range(self.model.H):
            for j in range(self.model.W):
                for k in range(self.model.L):
                    if i==0 or i==self.model.H-1 or j==0 or j==self.model.W-1 or k==0 or k==self.model.L-1:
                        self.surface_indices.append((i,j,k))


        # Initialize Pygame + OpenGL
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Ising Model - Spin Dynamics")

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(COLOR_BACKGROUND[0], COLOR_BACKGROUND[1], COLOR_BACKGROUND[2], 1.0)

        self.setup_projection()
        self.frame_count = 0

        # Initialize shaders & cube
        self.shader_program = self.compile_shaders()
        self.VAO, self.num_indices = self.create_cube_vao()
        self.instance_positions, self.instance_colors = self.get_surface_instances()
        self.update_instance_buffers()

    def perspective(self, fov, aspect, near, far):
        f = 1.0 / math.tan(fov / 2)
        return np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)

    def translate(self, x, y, z):
        return np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    def rotate_y(self, angle):
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [ c, 0, s, 0],
            [ 0, 1, 0, 0],
            [-s, 0, c, 0],
            [ 0, 0, 0, 1]
        ], dtype=np.float32)

    def setup_projection(self):
        glViewport(0, 100, WINDOW_WIDTH, WINDOW_HEIGHT - 100)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(100.0, float(WINDOW_WIDTH) / float(WINDOW_HEIGHT), 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)

    def compile_shaders(self):
        vertex_src = """
        #version 330 core

        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 iPos;
        layout (location = 2) in vec3 iColor;

        uniform mat4 MVP;

        out vec3 vColor;

        void main()
        {
            vec3 worldPos = aPos + iPos;
            gl_Position = MVP * vec4(worldPos, 1.0);
            vColor = iColor;
        }
        """
        fragment_src = """
        #version 330 core

        in vec3 vColor;
        out vec4 FragColor;

        void main()
        {
            FragColor = vec4(vColor, 1.0);
        }
        """
        vertex = compileShader(vertex_src, GL_VERTEX_SHADER)
        fragment = compileShader(fragment_src, GL_FRAGMENT_SHADER)
        return compileProgram(vertex, fragment)

    def create_cube_vao(self):
        s = self.voxel_size / 2
        vertices = np.array([
            -s, -s,  s,
            s, -s,  s,
            s,  s,  s,
            -s,  s,  s,
            -s, -s, -s,
            s, -s, -s,
            s,  s, -s,
            -s,  s, -s,
        ], dtype=np.float32)


        indices = np.array([
            0, 1, 2, 2, 3, 0,
            4, 6, 5, 4, 7, 6,
            4, 0, 3, 3, 7, 4,
            1, 5, 6, 6, 2, 1,
            3, 2, 6, 6, 7, 3,
            4, 5, 1, 1, 0, 4,
        ], dtype=np.uint32)

        VAO = glGenVertexArrays(1)
        glBindVertexArray(VAO)

        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        return VAO, len(indices)

    def get_surface_instances(self):
        """Return positions and colors of surface cubes"""
        positions, colors = [], []
        for i in range(self.model.H):
            for j in range(self.model.W):
                for k in range(self.model.L):
                    if i==0 or i==self.model.H-1 or j==0 or j==self.model.W-1 or k==0 or k==self.model.L-1:
                        x = (i - self.model.H/2) * self.voxel_size + self.voxel_size/2
                        y = (j - self.model.W/2) * self.voxel_size + self.voxel_size/2
                        z = (k - self.model.L/2) * self.voxel_size + self.voxel_size/2
                        positions.append([x, y, z])

                        spin = self.model.spins[i,j,k]
                        color = COLOR_SPIN_UP if spin==1 else COLOR_SPIN_DOWN
                        colors.append(color)
        return np.array(positions, dtype=np.float32), np.array(colors, dtype=np.float32)

    def update_instance_buffers(self):
        """Create/update GPU buffers for instance data"""
        glBindVertexArray(self.VAO)

        # Positions
        self.instanceVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instanceVBO)
        glBufferData(GL_ARRAY_BUFFER, self.instance_positions.nbytes, self.instance_positions, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)

        # Colors
        self.colorVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.colorVBO)
        glBufferData(GL_ARRAY_BUFFER, self.instance_colors.nbytes, self.instance_colors, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def update_colors(self):
        """Update instance colors from spins"""
        for idx, (i,j,k) in enumerate(self.surface_indices):
            self.instance_colors[idx] = (
                COLOR_SPIN_UP if self.model.spins[i,j,k] == 1 else COLOR_SPIN_DOWN
            )

        glBindBuffer(GL_ARRAY_BUFFER, self.colorVBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.instance_colors.nbytes, self.instance_colors)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render_3d_lattice(self):
        glUseProgram(self.shader_program)

        aspect = WINDOW_WIDTH / WINDOW_HEIGHT
        proj = self.perspective(math.radians(100), aspect, 0.1, 1000.0)

        camera_dist = 2.5 * self.lattice_radius
        view = self.translate(0, 0, -camera_dist)

        angle = time.time() * 0.4
        model = self.rotate_y(angle)

        MVP = proj @ view @ model

        loc = glGetUniformLocation(self.shader_program, "MVP")
        glUniformMatrix4fv(loc, 1, GL_TRUE, MVP)

        glBindVertexArray(self.VAO)
        glDrawElementsInstanced(
            GL_TRIANGLES,
            self.num_indices,
            GL_UNSIGNED_INT,
            None,
            len(self.instance_positions)
        )
        glBindVertexArray(0)

        glUseProgram(0)

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # ---------------------
        # 3D main lattice
        # ---------------------
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.render_3d_lattice()

        pygame.display.flip()

    

# ==============================================================================
# MAIN SIMULATION
# ==============================================================================
class IsingSim:
    """Main simulation controller"""
    
    def __init__(self):
        self.model = IsingModel3D(
            H=LATTICE_H,
            W=LATTICE_W,
            L=LATTICE_L,
            temperature=TEMPERATURE,
            j_coupling=J_COUPLING,
            external_field=EXTERNAL_FIELD
        )
        self.analyzer = PolarizationAnalyzer()
        self.visualizer = IsingSpin3DVisualizer(self.model, self.analyzer)
        
        self.running = True
        self.paused = False
        self.frame_count = 0
        self.start_time = time.time()

        # For decoupled simulation
        self.last_update_time = time.time()       # first timestamp
        self.simulation_accumulator = 0.0        # accumulated dt
        self.simulation_rate = 1000              # MC steps per second
        
    def handle_input(self):
        """Handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    # Reset model
                    self.model = IsingModel3D(
                        H=LATTICE_H,
                        W=LATTICE_W,
                        L=LATTICE_L,
                        temperature=TEMPERATURE,
                        j_coupling=J_COUPLING,
                        external_field=EXTERNAL_FIELD
                    )
                    # Update visualizer reference
                    self.visualizer.model = self.model
                elif event.key == pygame.K_t:
                    # Increase temperature
                    self.model.temperature *= 1.1
                    print(f"Temperature: {self.model.temperature:.2f}")
                elif event.key == pygame.K_y:
                    # Decrease temperature
                    self.model.temperature /= 1.1
                    print(f"Temperature: {self.model.temperature:.2f}")
    
    def update(self):
        """Update simulation"""
        if not self.paused:
            # Run enough steps based on time accumulator
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time

            self.simulation_accumulator += dt
            steps_to_run = int(self.simulation_accumulator * self.simulation_rate)
            if steps_to_run > 0:
                self.simulation_accumulator -= steps_to_run / self.simulation_rate
                self.model.run_steps(steps_to_run)
                self.visualizer.update_colors()

                # -------------------------------
                # Print statistics every N steps
                # -------------------------------
                if self.model.step_count // steps_to_run * steps_to_run % DISPLAY_UPDATE_INTERVAL == 0:
                    energy, magnetization = self.model.get_statistics()

                    # Analyze domains
                    domains = self.analyzer.find_domains(self.model.spins)
                    domain_stats = self.analyzer.get_domain_statistics(domains)

                    # Print statistics
                    print(f"\n{'='*60}")
                    print(f"MC Steps: {self.model.step_count}")
                    print(f"Temperature: {self.model.temperature:.2f}")
                    print(f"Energy: {energy:.4f}")
                    print(f"Magnetization: {magnetization:.4f}")
                    print(f"Polarization Magnitude: {self.analyzer.get_polarization_magnitude(self.model.spins):.4f}")
                    print(f"\nDomain Statistics:")
                    for key, value in domain_stats.items():
                        if isinstance(value, float):
                            print(f"  {key}: {value:.2f}")
                        else:
                            print(f"  {key}: {value}")

        self.frame_count += 1

    
    def render(self):
        """Render visualization"""
        self.visualizer.render()
    
    def run(self):
        """Main simulation loop"""
        
        print("3D Ising Model Simulator")
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  R: Reset")
        print("  T: Increase temperature")
        print("  Y: Decrease temperature")
        print("  ESC: Quit")
        
        clock = pygame.time.Clock()
        while self.running:
            self.handle_input()
            self.update()
            self.render()
        
        pygame.quit()

# ==============================================================================
# RUN SIMULATION
# ==============================================================================
if __name__ == "__main__":
    import ctypes
    sim = IsingSim()
    sim.run()
