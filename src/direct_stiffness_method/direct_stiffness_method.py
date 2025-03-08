import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eigh

class Structure:
    def __init__(self, nodes, elements, element_properties):
        self.nodes = nodes
        self.elements = elements
        self.element_properties = element_properties
        self.E = {elem_id: props["E"] for elem_id, props in element_properties.items()}
        self.nu = {elem_id: props["nu"] for elem_id, props in element_properties.items()}
    
    def compute_section_properties(self, elem_id):
        """Compute A, Iy, Iz, J for a given element based on its shape."""
        elem = self.element_properties[elem_id]

        if "b" in elem and "h" in elem:  # Rectangular section
            b = elem["b"]
            h = elem["h"]
            A = b * h
            Iy = (h * b**3) / 12
            Iz = (b * h**3) / 12
            J = Iy + Iz

        elif "r" in elem:  # Circular section
            r = elem["r"]
            A = math.pi * r**2
            Iy = Iz = (math.pi * r**4) / 4
            J = Iy + Iz

        else:
            raise ValueError("Invalid element properties. Define either (b, h) or (r).")

        return A, Iy, Iz, J
        
    def element_length(self, node1, node2):
        x1, y1, z1 = self.nodes[node1][1:]
        x2, y2, z2 = self.nodes[node2][1:]
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    def local_elastic_stiffness_matrix_3D_beam(self, elem_id, L):
        """
        Compute the local element elastic stiffness matrix for a 3D beam.
        """
        k_e = np.zeros((12, 12))

        # Retrieve individual properties
        E = self.E[elem_id]
        nu = self.nu[elem_id]
        A, Iy, Iz, J = self.compute_section_properties(elem_id)

        # Axial terms - extension of local x axis
        axial_stiffness = E * A / L
        k_e[0, 0] = axial_stiffness
        k_e[0, 6] = -axial_stiffness
        k_e[6, 0] = -axial_stiffness
        k_e[6, 6] = axial_stiffness

        # Torsion terms - rotation about local x axis
        torsional_stiffness = E * J / (2.0 * (1 + nu) * L)
        k_e[3, 3] = torsional_stiffness
        k_e[3, 9] = -torsional_stiffness
        k_e[9, 3] = -torsional_stiffness
        k_e[9, 9] = torsional_stiffness

        # Bending terms - bending about local z axis
        k_e[1, 1] = E * 12.0 * Iz / L ** 3.0
        k_e[1, 7] = E * -12.0 * Iz / L ** 3.0
        k_e[7, 1] = E * -12.0 * Iz / L ** 3.0
        k_e[7, 7] = E * 12.0 * Iz / L ** 3.0
        k_e[1, 5] = E * 6.0 * Iz / L ** 2.0
        k_e[5, 1] = E * 6.0 * Iz / L ** 2.0
        k_e[1, 11] = E * 6.0 * Iz / L ** 2.0
        k_e[11, 1] = E * 6.0 * Iz / L ** 2.0
        k_e[5, 7] = E * -6.0 * Iz / L ** 2.0
        k_e[7, 5] = E * -6.0 * Iz / L ** 2.0
        k_e[7, 11] = E * -6.0 * Iz / L ** 2.0
        k_e[11, 7] = E * -6.0 * Iz / L ** 2.0
        k_e[5, 5] = E * 4.0 * Iz / L
        k_e[11, 11] = E * 4.0 * Iz / L
        k_e[5, 11] = E * 2.0 * Iz / L
        k_e[11, 5] = E * 2.0 * Iz / L

        # Bending terms - bending about local y axis
        k_e[2, 2] = E * 12.0 * Iy / L ** 3.0
        k_e[2, 8] = E * -12.0 * Iy / L ** 3.0
        k_e[8, 2] = E * -12.0 * Iy / L ** 3.0
        k_e[8, 8] = E * 12.0 * Iy / L ** 3.0
        k_e[2, 4] = E * -6.0 * Iy / L ** 2.0
        k_e[4, 2] = E * -6.0 * Iy / L ** 2.0
        k_e[2, 10] = E * -6.0 * Iy / L ** 2.0
        k_e[10, 2] = E * -6.0 * Iy / L ** 2.0
        k_e[4, 8] = E * 6.0 * Iy / L ** 2.0
        k_e[8, 4] = E * 6.0 * Iy / L ** 2.0
        k_e[8, 10] = E * 6.0 * Iy / L ** 2.0
        k_e[10, 8] = E * 6.0 * Iy / L ** 2.0
        k_e[4, 4] = E * 4.0 * Iy / L
        k_e[10, 10] = E * 4.0 * Iy / L
        k_e[4, 10] = E * 2.0 * Iy / L
        k_e[10, 4] = E * 2.0 * Iy / L

        return k_e

    def compute_local_stiffness_matrices(self):
        """
        Computes and stores local stiffness matrices for all elements.
        """
        stiffness_matrices = {}
        for i, (n1, n2) in enumerate(self.elements):
            L = self.element_length(n1, n2)
            stiffness_matrices[i] = self.local_elastic_stiffness_matrix_3D_beam(i, L)
        return stiffness_matrices
    
    def compute_global_stiffness_matrix(self):
        """Compute and assemble the global stiffness matrix."""
        local_stiffness_matrices = self.compute_local_stiffness_matrices()
        stiffness_handler = StiffnessMatrices(self)
        
        # Map to global stiffness matrices
        global_stiffness_matrices = stiffness_handler.compute_global_stiffness_matrices(local_stiffness_matrices)
        
        # Assemble the final global stiffness matrix
        return stiffness_handler.assemble_global_stiffness_matrix(global_stiffness_matrices)
    
class StiffnessMatrices:
    def __init__(self, structure):
        self.structure = structure
    
    def compute_global_stiffness_matrices(self, local_stiffness_matrices):
        """
        Computes the global stiffness matrices by mapping local stiffness matrices from local to global coordinates.
        """
        global_stiffness_matrices = {}
        
        for i, (n1, n2) in enumerate(self.structure.elements):
            # Obtain nodal coordinates
            x1, y1, z1 = self.structure.nodes[n1][1:]
            x2, y2, z2 = self.structure.nodes[n2][1:]

            # Compute the 3x3 rotation matrix
            gamma = rotation_matrix_3D(x1, y1, z1, x2, y2, z2)

            # Compute the 12x12 transformation matrix
            Gamma = transformation_matrix_3D(gamma)

            # Transform the local stiffness matrix to global coordinates
            k_local = local_stiffness_matrices[i]
            k_global = Gamma.T @ k_local @ Gamma
            global_stiffness_matrices[i] = k_global

        return global_stiffness_matrices
    
    def assemble_global_stiffness_matrix(self, global_stiffness_matrices):
        """
        Assembles the global stiffness matrix from element stiffness matrices.
        """
        n_global_nodes = len(self.structure.nodes)
        total_dofs = n_global_nodes * 6
        K_global_assembled = np.zeros((total_dofs, total_dofs))

        for elem_idx, (node1, node2) in enumerate(self.structure.elements):
            # Global DOF indices for the element
            dofs = np.r_[node1 * 6 : node1 * 6 + 6, node2 * 6 : node2 * 6 + 6]

            # Assemble element stiffness into global matrix
            k_global = global_stiffness_matrices[elem_idx]
            K_global_assembled[np.ix_(dofs, dofs)] += k_global

        return K_global_assembled
    
class BoundaryConditions:
    def __init__(self, loads, supports):
        """
        Initializes the boundary conditions with loads and supports.
        """
        self.loads = loads
        self.supports = supports
        self.n_nodes = len(loads)  # Assuming all nodes have loads/supports defined

    def compute_global_load_vector(self):
        """
        Constructs and returns the global load vector as a column vector.
        """
        total_dofs = self.n_nodes * 6
        F_global = np.zeros((total_dofs, 1))  # Column vector (total_dofs x 1)

        for node, values in self.loads.items():
            dof_index = node * 6  # Starting DOF index for the node
            F_global[dof_index:dof_index + 6, 0] = values[1:]  # Skip the node number
                
        return F_global

class Solver:
    def __init__(self, structure, boundary_conditions):
        """
        Initializes the solver with the structure and boundary conditions.
        """
        self.structure = structure
        self.boundary_conditions = boundary_conditions
    
    def solve(self):
        """
        Solves for the unknown displacements by reducing the global system and solving:
        U_reduced = K_reduced^-1 * F_reduced.
        """
        # Step 1: Assemble global matrices
        K_global = self.structure.compute_global_stiffness_matrix()
        F_global = self.boundary_conditions.compute_global_load_vector()

        # Step 2: Identify constrained DOFs directly
        constrained_dofs = {node * 6 + dof for node, constraints in self.boundary_conditions.supports.items()
                            for dof in range(6) if constraints[dof + 1] == 1}
        free_dofs = np.setdiff1d(np.arange(K_global.shape[0]), list(constrained_dofs))

        # Step 3: Extract the reduced system
        K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
        F_reduced = F_global[free_dofs]

        # Step 4: Solve for unknown displacements
        U_reduced = np.linalg.solve(K_reduced, F_reduced)

        # Step 5: Reassemble full displacement vector
        U_global = np.zeros_like(F_global)
        U_global[free_dofs] = U_reduced

        return U_global
    
    def compute_reactions(self, U_global):
        """
        Computes reaction forces at constrained degrees of freedom.
        """
        K_global = self.structure.compute_global_stiffness_matrix()
        constrained_dofs = {node * 6 + dof for node, constraints in self.boundary_conditions.supports.items()
                            for dof in range(6) if constraints[dof + 1] == 1}
        
        # Compute reaction forces at constrained DOFs
        R_global = np.zeros_like(U_global)
        R_global[list(constrained_dofs)] = K_global[list(constrained_dofs)] @ U_global

        return R_global

class BucklingAnalysis:
    def __init__(self, structure, solver, U_global):
        """
        Initializes the buckling analysis with the structure and computed internal forces.
        """
        self.structure = structure
        self.U_global = U_global
        self.internal_forces = self.compute_internal_forces()

    def compute_internal_forces(self):
        """Compute internal forces for each element."""
        internal_forces = {}

        for elem_id, (node_i, node_j) in enumerate(self.structure.elements):
            # Extract global displacements for the element
            Ue_global = np.hstack([
                self.U_global[node_i * 6:(node_i + 1) * 6].flatten(),
                self.U_global[node_j * 6:(node_j + 1) * 6].flatten()
            ])

            # Compute the transformation matrix
            x1, y1, z1 = self.structure.nodes[node_i][1:]
            x2, y2, z2 = self.structure.nodes[node_j][1:]
            gamma = rotation_matrix_3D(x1, y1, z1, x2, y2, z2)
            T = transformation_matrix_3D(gamma)

            # Transform to local coordinates
            Ue_local = T @ Ue_global

            # Retrieve local stiffness matrix
            L = self.structure.element_length(node_i, node_j)
            k_local = self.structure.local_elastic_stiffness_matrix_3D_beam(elem_id, L)

            # Compute internal forces in local coordinates
            internal_forces[elem_id] = k_local @ Ue_local
        
        return internal_forces

    def local_geometric_stiffness_matrix_3D_beam(self, elem_id, L):
        """
        Compute the local element geometric stiffness matrix for a 3D beam.
        """
        k_g = np.zeros((12, 12))

        # Retrieve individual properties
        A, Iy, Iz, J = self.structure.compute_section_properties(elem_id)

        forces = self.internal_forces[elem_id]
        Fx2 = forces[6]
        Mx2 = forces[9]
        My1 = forces[4]
        Mz1 = forces[5]
        My2 = forces[10]
        Mz2 = forces[11]

        # upper triangle off diagonal terms
        k_g[0, 6] = -Fx2 / L
        k_g[1, 3] = My1 / L
        k_g[1, 4] = Mx2 / L
        k_g[1, 5] = Fx2 / 10.0
        k_g[1, 7] = -6.0 * Fx2 / (5.0 * L)
        k_g[1, 9] = My2 / L
        k_g[1, 10] = -Mx2 / L
        k_g[1, 11] = Fx2 / 10.0
        k_g[2, 3] = Mz1 / L
        k_g[2, 4] = -Fx2 / 10.0
        k_g[2, 5] = Mx2 / L
        k_g[2, 8] = -6.0 * Fx2 / (5.0 * L)
        k_g[2, 9] = Mz2 / L
        k_g[2, 10] = -Fx2 / 10.0
        k_g[2, 11] = -Mx2 / L
        k_g[3, 4] = -1.0 * (2.0 * Mz1 - Mz2) / 6.0
        k_g[3, 5] = (2.0 * My1 - My2) / 6.0
        k_g[3, 7] = -My1 / L
        k_g[3, 8] = -Mz1 / L
        k_g[3, 9] = -Fx2 * J / (A * L)
        k_g[3, 10] = -1.0 * (Mz1 + Mz2) / 6.0
        k_g[3, 11] = (My1 + My2) / 6.0
        k_g[4, 7] = -Mx2 / L
        k_g[4, 8] = Fx2 / 10.0
        k_g[4, 9] = -1.0 * (Mz1 + Mz2) / 6.0
        k_g[4, 10] = -Fx2 * L / 30.0
        k_g[4, 11] = Mx2 / 2.0
        k_g[5, 7] = -Fx2 / 10.0
        k_g[5, 8] = -Mx2 / L
        k_g[5, 9] = (My1 + My2) / 6.0
        k_g[5, 10] = -Mx2 / 2.0
        k_g[5, 11] = -Fx2 * L / 30.0
        k_g[7, 9] = -My2 / L
        k_g[7, 10] = Mx2 / L
        k_g[7, 11] = -Fx2 / 10.0
        k_g[8, 9] = -Mz2 / L
        k_g[8, 10] = Fx2 / 10.0
        k_g[8, 11] = Mx2 / L
        k_g[9, 10] = (Mz1 - 2.0 * Mz2) / 6.0
        k_g[9, 11] = -1.0 * (My1 - 2.0 * My2) / 6.0

        # add in the symmetric lower triangle
        k_g = k_g + k_g.transpose()

        # add diagonal terms
        k_g[0, 0] = Fx2 / L
        k_g[1, 1] = 6.0 * Fx2 / (5.0 * L)
        k_g[2, 2] = 6.0 * Fx2 / (5.0 * L)
        k_g[3, 3] = Fx2 * J / (A * L)
        k_g[4, 4] = 2.0 * Fx2 * L / 15.0
        k_g[5, 5] = 2.0 * Fx2 * L / 15.0
        k_g[6, 6] = Fx2 / L
        k_g[7, 7] = 6.0 * Fx2 / (5.0 * L)
        k_g[8, 8] = 6.0 * Fx2 / (5.0 * L)
        k_g[9, 9] = Fx2 * J / (A * L)
        k_g[10, 10] = 2.0 * Fx2 * L / 15.0
        k_g[11, 11] = 2.0 * Fx2 * L / 15.0

        return k_g

    def compute_local_geometric_stiffness_matrices(self):
        """
        Computes and stores local geometric stiffness matrices for all elements.
        """
        geometric_stiffness_matrices = {}
        for i, (n1, n2) in enumerate(self.structure.elements):
            L = self.structure.element_length(n1, n2)
            geometric_stiffness_matrices[i] = self.local_geometric_stiffness_matrix_3D_beam(i, L)
        return geometric_stiffness_matrices
    
    def compute_global_geometric_stiffness_matrix(self):
        """Compute and assemble the global geometric stiffness matrix."""
        local_geometric_stiffness_matrices = self.compute_local_geometric_stiffness_matrices()
        stiffness_handler = StiffnessMatrices(self.structure)  # FIXED! Pass structure

        # Map to global stiffness matrices
        global_geometric_stiffness_matrices = stiffness_handler.compute_global_stiffness_matrices(local_geometric_stiffness_matrices)

        # Assemble the final global stiffness matrix
        return stiffness_handler.assemble_global_stiffness_matrix(global_geometric_stiffness_matrices)

    def calculate_critical_load(self, solver):
        """
        Calculates the critical load and reconstructs the full global mode shape.
        
        Returns:
            - min_eigenvalue: The minimum positive eigenvalue (critical buckling load).
            - global_mode_shape: The full eigenvector (mode shape) with zeros at constrained DOFs.
        """
        # Compute the global stiffness matrices
        Kg_global_assembled = self.compute_global_geometric_stiffness_matrix()
        K_global_assembled = self.structure.compute_global_stiffness_matrix()

        # Identify constrained DOFs directly
        constrained_dofs = {node * 6 + dof for node, constraints in solver.boundary_conditions.supports.items()
                            for dof in range(6) if constraints[dof + 1] == 1}
        free_dofs = np.setdiff1d(np.arange(K_global_assembled.shape[0]), list(constrained_dofs))

        # Extract submatrices for free DOFs
        K_reduced = K_global_assembled[np.ix_(free_dofs, free_dofs)]
        Kg_reduced = Kg_global_assembled[np.ix_(free_dofs, free_dofs)]

        # Solve the generalized eigenvalue problem
        eigenvalues, eigenvectors = eigh(K_reduced, -Kg_reduced)

        # Extract the smallest positive eigenvalue
        positive_eigenvalues = eigenvalues[eigenvalues > 0]
        
        if len(positive_eigenvalues) == 0:
            raise ValueError("No positive eigenvalues found. Check the system setup.")

        min_eigenvalue = np.min(positive_eigenvalues)

        # Find the index of the minimum positive eigenvalue
        min_index = np.where(eigenvalues == min_eigenvalue)[0][0]

        # Extract the corresponding reduced eigenvector
        min_eigenvector = eigenvectors[:, min_index]

        # Normalize the eigenvector to ensure consistent magnitude
        min_eigenvector /= np.linalg.norm(min_eigenvector)

        # Create the full global mode shape and insert zeros at constrained DOFs
        global_mode_shape = np.zeros(K_global_assembled.shape[0])
        global_mode_shape[free_dofs] = min_eigenvector  # Fill only free DOFs

        return min_eigenvalue, global_mode_shape



class PlotResults:
    def __init__(self, structure, U_global, scale=1.0):
        """
        Initializes the plotting class with the structure and deformation data.

        Parameters:
        - structure: Structure object containing nodes and elements.
        - U_global: Global displacement vector (translations & rotations).
        - scale: Scaling factor for visualization.
        """
        self.structure = structure
        self.U_global = U_global
        self.scale = scale  # Scaling factor for deformation

    def hermite_interpolation(self, u1, theta1, u2, theta2, L, num_points=20):
        """
        Performs Hermite cubic interpolation for bending displacements using the exact
        shape functions from the provided image.

        Parameters:
        - u1, u2: Displacements at both nodes.
        - theta1, theta2: Rotations at both nodes.
        - L: Element length.
        - num_points: Number of interpolation points.

        Returns:
        - Interpolated displacement values.
        """
        x_vals = np.linspace(0, L, num_points)  # Physical coordinate x
        
        # Shape functions from the image
        N1 = 1 - 3 * (x_vals / L) ** 2 + 2 * (x_vals / L) ** 3
        N2 = 3 * (x_vals / L) ** 2 - 2 * (x_vals / L) ** 3
        N3 = x_vals * (1 - x_vals / L) ** 2
        N4 = x_vals * ((x_vals / L) ** 2 - x_vals / L)

        # Compute interpolated displacements
        u_interp = N1 * u1 + N2 * u2 + N3 * theta1 + N4 * theta2
        return u_interp


    def plot_deformed_shape(self, num_points=20):
        """
        Plots the deformed shape of the 3D beam structure using Hermite interpolation,
        including transformation to local coordinates for elements in arbitrary directions.
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot undeformed shape (dashed black lines)
        for (n1, n2) in self.structure.elements:
            node1_coords = np.array(self.structure.nodes[n1][1:])
            node2_coords = np.array(self.structure.nodes[n2][1:])

            ax.plot([node1_coords[0], node2_coords[0]],
                    [node1_coords[1], node2_coords[1]],
                    [node1_coords[2], node2_coords[2]], 'k--', linewidth=0.5)  # Dashed black for undeformed

        # Plot deformed shape (interpolated)
        for i, (n1, n2) in enumerate(self.structure.elements):
            # Get original nodal coordinates
            x1, y1, z1 = self.structure.nodes[n1][1:]
            x2, y2, z2 = self.structure.nodes[n2][1:]

            # Compute element length
            L = self.structure.element_length(n1, n2)

            # Compute rotation matrix for transformation
            gamma = rotation_matrix_3D(x1, y1, z1, x2, y2, z2)
            Gamma = transformation_matrix_3D(gamma)  # 12x12 transformation matrix

            # Extract global displacements & rotations
            U_global_element = np.hstack([
                self.U_global[n1 * 6:(n1 * 6) + 6].flatten(),
                self.U_global[n2 * 6:(n2 * 6) + 6].flatten()
            ])

            # Transform to local coordinates
            U_local_element = Gamma @ U_global_element

            # Extract local displacements
            u1, v1, w1, theta_x1, theta_y1, theta_z1 = U_local_element[:6]
            u2, v2, w2, theta_x2, theta_y2, theta_z2 = U_local_element[6:]

            # Compute axial displacement (linear interpolation)
            u_interp_local_x = np.linspace(u1, u2, num_points) * self.scale

            # Bending in XY plane (v-displacement) is associated with θz
            u_interp_local_y = self.hermite_interpolation(v1, theta_z1, v2, theta_z2, L, num_points) * self.scale
            # Bending in XZ plane (w-displacement) is associated with θy
            u_interp_local_z = self.hermite_interpolation(w1, theta_y1, w2, theta_y2, L, num_points) * self.scale

            # Stack into an array for transformation back to global coordinates
            u_interp_local = np.vstack([u_interp_local_x, u_interp_local_y, u_interp_local_z])

            # # Transform back to global coordinates
            u_interp_global = gamma.T @ u_interp_local
            
            # Compute deformed positions
            x_interp = np.linspace(x1, x2, num_points) + u_interp_global[0, :]
            y_interp = np.linspace(y1, y2, num_points) + u_interp_global[1, :]
            z_interp = np.linspace(z1, z2, num_points) + u_interp_global[2, :]

            # Plot deformed element
            ax.plot(x_interp, y_interp, z_interp, 'r', linewidth=1.5)

        # Plot nodes
        node_coords = np.array([self.structure.nodes[n][1:] for n in self.structure.nodes])
        ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], color='b', s=20, label="Nodes")

        # Labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title('Deformed Structure (Red) vs Undeformed (Dashed Black)')
        plt.legend()
        plt.show()



# Useful functions
def rotation_matrix_3D(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, v_temp: np.ndarray = None):
    """
    3D rotation matrix
    source: Chapter 5.1 of McGuire's Matrix Structural Analysis 2nd Edition
    Given:
        x, y, z coordinates of the ends of two beams: x1, y1, z1, x2, y2, z2
        optional: reference vector v_temp to orthonormalize the local y and z axes.
            If v_temp is not provided, a default is chosen based on the beam orientation.
    Compute:
        A 3x3 rotation matrix where the rows represent the local x, y, and z axes in global coordinates.
    """
    L = np.sqrt((x2 - x1) ** 2.0 + (y2 - y1) ** 2.0 + (z2 - z1) ** 2.0)
    lxp = (x2 - x1) / L
    mxp = (y2 - y1) / L
    nxp = (z2 - z1) / L
    local_x = np.asarray([lxp, mxp, nxp])

    # Choose a vector to orthonormalize the local y axis if one is not given
    if v_temp is None:
        # if the beam is oriented vertically, switch to the global y axis
        if np.isclose(lxp, 0.0) and np.isclose(mxp, 0.0):
            v_temp = np.array([0, 1.0, 0.0])
        else:
            # otherwise use the global z axis
            v_temp = np.array([0, 0, 1.0])

    # Compute the local y axis by taking the cross product of v_temp and local_x
    local_y = np.cross(v_temp, local_x)
    local_y = local_y / np.linalg.norm(local_y)

    # Compute the local z axis
    local_z = np.cross(local_x, local_y)
    local_z = local_z / np.linalg.norm(local_z)

    # Assemble the rotation matrix (gamma)
    gamma = np.vstack((local_x, local_y, local_z))
    
    return gamma

def transformation_matrix_3D(gamma: np.ndarray) -> np.ndarray:
    """
    3D transformation matrix
    source: Chapter 5.1 of McGuire's Matrix Structural Analysis 2nd Edition
    Given:
        gamma -- the 3x3 rotation matrix
    Compute:
        Gamma -- the 12x12 transformation matrix which maps the 6 DOFs at each node.
    """
    Gamma = np.zeros((12, 12))
    Gamma[0:3, 0:3] = gamma
    Gamma[3:6, 3:6] = gamma
    Gamma[6:9, 6:9] = gamma
    Gamma[9:12, 9:12] = gamma
    return Gamma