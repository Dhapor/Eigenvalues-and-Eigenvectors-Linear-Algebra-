import streamlit as st
import numpy as np
import sympy as sp
from sympy import Matrix, symbols, Eq, Rational, Poly

st.set_page_config(page_title="Linear Algebra Calculator", layout="centered")
st.title("🦮 Linear Algebra Calculator")
st.markdown("---")

# Session state for matrix memory
if "matrix_input" not in st.session_state:
    st.session_state.matrix_input = "1 3\n2 -4"

x = symbols('x')

# Sidebar structure
st.sidebar.title("Main Sections")
main_section = st.sidebar.radio("Navigate to:", ["Home", "Tutorial", "Features"])

if main_section == "Features":
    st.sidebar.markdown("---")
    feature_page = st.sidebar.radio("Select Feature", [
        "Eigenvalues & Eigenvectors",
        "Diagonalization",
        "Characteristic Polynomial",
        "Minimal Polynomial",
        "Cayley-Hamilton Theorem",
        "Gram-Schmidt Process",
        "Build Matrix from Eigenvectors",
        "Jordan Canonical Form"
    ])
else:
    feature_page = None

def input_matrix(label="Enter matrix A"):
    raw = st.text_area(label, st.session_state.matrix_input)
    st.session_state.matrix_input = raw
    try:
        lines = raw.strip().split("\n")
        matrix = [[sp.sympify(entry) for entry in line.strip().split()] for line in lines]
        return Matrix(matrix)
    except:
        st.error("Invalid matrix format. Use space-separated numbers or rational expressions like 1/2.")
        return None

def simplify_expr(expr):
    return sp.nsimplify(expr, rational=True)

# Home Page
if main_section == "Home":
    st.header("🏠 Welcome to the Linear Algebra Calculator")
    st.image("a.jpg", use_column_width=True)  # This displays the image full-width
    st.markdown("""
    This web application is designed to help you explore and understand key concepts in **Linear Algebra**.

    ### 🔍 What You Can Do:
    - Compute **Eigenvalues and Eigenvectors**
    - Perform **Matrix Diagonalization**
    - Get the **Characteristic Polynomial**
    - Compute the **Minimal Polynomial**
    - Apply the **Cayley-Hamilton Theorem**
    - Apply the **Gram-Schmidt Process** for orthonormal bases
    - Rebuild a matrix from given eigenvectors and eigenvalues
    - Compute **Jordan Canonical Form**

    Built with ❤️ by **Datapsalm** and the **PENTAGON** (Math 300L Squad)
    """)

# Tutorial Page
elif main_section == "Tutorial":
    st.header("📘 How to Use This App")
    st.markdown("""
    ### 1⃣ Matrix Input Format
    Enter your matrix in the text area like this:
    ```
    1 3
    2 -4
    ```
    - Each row is a new line
    - Separate entries with spaces
    - You can use fractions (e.g., `1/2`) or integers

    ### 2⃣ Features
    - Go to the sidebar and select **Features**, then choose a topic.
    - Input your matrix.
    - Click the compute button for the selected operation.

    ### 3⃣ Output
    - Results will appear below the button.
    - Expressions are simplified to clean fractions or integers.

    ### 💡 Tips
    - Always ensure the matrix is **square** when computing eigenvalues or diagonalizing.
    - You can reuse the same matrix across different operations without retyping it.
    - Use the 'Build Matrix from Eigenvectors' tool to reverse-engineer a matrix given its eigen components.
    """)

# Features Section
elif main_section == "Features" and feature_page:
    if feature_page == "Eigenvalues & Eigenvectors":
        st.header("🔍 Eigenvalues & Eigenvectors")
        A = input_matrix("Enter square matrix A:")
        if st.button("🔎 Compute Eigenvalues & Eigenvectors"):
            if A is not None:
                if A.shape[0] != A.shape[1]:
                    st.error("Matrix must be square")
                else:
                    eigs = A.eigenvals()
                    st.subheader("Eigenvalues")
                    for val, mult in eigs.items():
                        st.latex(f"\\lambda = {sp.latex(simplify_expr(val))}, \\text{{Multiplicity}} = {mult}")

                    st.subheader("Eigenvectors")
                    eigvecs = A.eigenvects()
                    for val, mult, basis in eigvecs:
                        st.write(f"Eigenvalue: {simplify_expr(val)}")
                        for vec in basis:
                            st.latex(sp.latex(vec.applyfunc(simplify_expr)))

    elif feature_page == "Diagonalization":
        st.header("📐 Diagonalization")
        A = input_matrix("Enter a diagonalizable square matrix A:")
        if st.button("📄 Diagonalize Matrix"):
            if A is not None:
                if A.shape[0] != A.shape[1]:
                    st.error("Matrix must be square")
                else:
                    try:
                        P, D = A.diagonalize()
                        st.subheader("Matrix P (Eigenvectors)")
                        st.latex(sp.latex(P.applyfunc(simplify_expr)))
                        st.subheader("Diagonal Matrix D")
                        st.latex(sp.latex(D.applyfunc(simplify_expr)))
                        st.info("A = P * D * P⁻¹")
                    except Exception as e:
                        st.error(f"Diagonalization failed: {e}")

    elif feature_page == "Characteristic Polynomial":
        st.header("🗞 Characteristic Polynomial")
        A = input_matrix("Enter square matrix A:")
        if st.button("🧲 Compute Characteristic Polynomial"):
            if A is not None:
                if A.shape[0] != A.shape[1]:
                    st.error("Matrix must be square")
                else:
                    I = sp.eye(A.shape[0])
                    char_poly = (A - x * I).det()
                    st.write("Characteristic Polynomial:")
                    st.latex(sp.latex(sp.simplify(char_poly)))

    elif feature_page == "Minimal Polynomial":
        st.header("🧾 Minimal Polynomial")
        A = input_matrix("Enter square matrix A:")
        if st.button("📉 Compute Minimal Polynomial"):
            try:
                λ = sp.symbols('λ')
                n = A.shape[0]
                if n != A.shape[1]:
                    st.error("Matrix must be square.")
                else:
                    # Characteristic polynomial
                    char_poly = (A - λ * sp.eye(n)).det()
                    factors = sp.factor_list(char_poly)[1]  # [(factor, exponent), ...]

                    from itertools import product
                    def build_poly(combo):
                        poly = 1
                        for (factor, _), exp in zip(factors, combo):
                            poly *= factor**exp
                        return sp.expand(poly)

                    # Try combinations of factor powers
                    power_ranges = [range(1, exp + 1) for (_, exp) in factors]
                    found = False

                    for combo in product(*power_ranges):
                        candidate = build_poly(combo)
                        coeffs = sp.Poly(candidate, λ).all_coeffs()
                        result = sp.zeros(*A.shape)
                        for i, c in enumerate(coeffs):
                            result += sp.sympify(c) * A**(len(coeffs) - i - 1)
                        if result == sp.zeros(*A.shape):
                            min_poly = candidate
                            found = True
                            break

                    if found:
                        st.subheader("✅ Minimal Polynomial:")
                        st.latex(sp.latex(sp.factor(min_poly)))
                    else:
                        st.warning("⚠️ Couldn't determine minimal polynomial.")
            except Exception as e:
                st.error(f"Error computing minimal polynomial: {e}")






    elif feature_page == "Cayley-Hamilton Theorem":
        st.header("📜 Cayley-Hamilton Theorem")
        A = input_matrix("Enter square matrix A:")
        if st.button("📘 Verify Cayley-Hamilton Theorem"):
            if A is not None:
                try:
                    n = A.shape[0]
                    I = sp.eye(n)
                    p = (A - x*I).det().expand()
                    st.subheader("Characteristic Polynomial:")
                    st.latex(sp.latex(p))
                    coeffs = list(reversed(p.as_poly(x).all_coeffs()))
                    substituted = sum((coeff * A**i for i, coeff in enumerate(coeffs)), start=sp.zeros(*A.shape))

                    simplified = substituted.applyfunc(sp.simplify)
                    st.subheader("Substituting into Polynomial (Should yield Zero Matrix):")
                    st.latex(sp.latex(simplified))
                    if simplified == sp.zeros(*A.shape):
                        st.success("✅ Cayley-Hamilton Theorem Verified!")
                    else:
                        st.warning("⚠️ Cayley-Hamilton Theorem NOT verified. Check your matrix.")
                except Exception as e:
                    st.error(f"Error: {e}")

    elif feature_page == "Gram-Schmidt Process":
        st.header("📐 Gram-Schmidt Orthonormalization")
        A = input_matrix("Enter matrix with vectors as rows:")
        if st.button("📏 Apply Gram-Schmidt"):
            if A is not None:
                try:
                    vectors = [A.row(i) for i in range(A.rows)]
                    ortho_basis = []
                    for v in vectors:
                        for u in ortho_basis:
                            v -= v.project(u)
                        ortho_basis.append(v)
                    ortho_basis_normalized = [vec.normalized() for vec in ortho_basis]

                    st.subheader("✅ Orthonormal Basis:")
                    for i, vec in enumerate(ortho_basis_normalized):
                        st.latex(f"\\vec{{v}}_{{{i+1}}} = {sp.latex(vec.applyfunc(simplify_expr))}")
                except Exception as e:
                    st.error(f"Failed to compute orthonormal basis: {e}")

    elif feature_page == "Build Matrix from Eigenvectors":
        st.header("🧱 Build Matrix from Eigenvectors & Eigenvalues")
        size = st.selectbox("Matrix Size:", [2, 3], index=1)

        st.markdown("### Step 1: Enter Eigenvectors")
        vectors = []
        for i in range(size):
            v_input = st.text_input(f"Eigenvector {i+1} (space-separated):", value=" ".join("1" if j == i else "0" for j in range(size)))
            try:
                v = Matrix([sp.sympify(x) for x in v_input.strip().split()])
                if len(v) != size:
                    st.error(f"Eigenvector {i+1} must have {size} components.")
                else:
                    vectors.append(v)
            except:
                st.error(f"Invalid input for eigenvector {i+1}.")

        st.markdown("### Step 2: Enter Corresponding Eigenvalues")
        lambdas = []
        for i in range(size):
            l = st.text_input(f"Eigenvalue for Vector {i+1}", value=str(i+1))
            try:
                lambdas.append(sp.sympify(l))
            except:
                st.error(f"Invalid eigenvalue {i+1}.")

        if st.button("🧠 Compute Matrix A = P D P⁻¹"):
            if len(vectors) == size and len(lambdas) == size:
                try:
                    P = Matrix.hstack(*vectors)
                    D = Matrix.diag(*lambdas)
                    A = P * D * P.inv()
                    st.subheader("✅ Reconstructed Matrix A:")
                    st.latex(sp.latex(A.applyfunc(simplify_expr)))
                except Exception as e:
                    st.error(f"Failed to compute matrix A: {e}")

    elif feature_page == "Jordan Canonical Form":
        st.header("🏛 Jordan Canonical Form")
        A = input_matrix("Enter square matrix A:")
        if st.button("📐 Compute Jordan Form"):
            if A is not None:
                try:
                    J, P = A.jordan_form(calc_transform=True)
                    st.subheader("Jordan Canonical Form J:")
                    st.latex(sp.latex(J.applyfunc(simplify_expr)))
                    st.subheader("Transformation Matrix P:")
                    st.latex(sp.latex(P.applyfunc(simplify_expr)))
                    st.info("A = P * J * P⁻¹")
                except Exception as e:
                    st.error(f"Could not compute Jordan form: {e}")

st.markdown("---")
st.markdown("<h5 style='text-align: center;'>Built with ❤️ by Datapsalm & the PENTAGON</h5>", unsafe_allow_html=True)
