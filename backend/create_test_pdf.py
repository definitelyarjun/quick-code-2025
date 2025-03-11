from fpdf import FPDF

# Create a PDF object
pdf = FPDF()

# Add a page
pdf.add_page()

# Set font
pdf.set_font("Arial", size=12)

# Add text
pdf.cell(200, 10, txt="UNIX Operating System Documentation", ln=True, align='C')
pdf.ln(10)
pdf.multi_cell(0, 10, txt="This is a test document about UNIX operating systems. UNIX is a family of multitasking, multiuser computer operating systems. The structure of UNIX consists of the kernel, shell, file system, and utilities. The kernel is the core of the operating system that manages resources. The shell is the command interpreter that provides the user interface. The file system organizes data on storage devices. Utilities are programs that perform specific tasks.")
pdf.ln(10)
pdf.multi_cell(0, 10, txt="The UNIX kernel is responsible for managing the computer's resources, such as CPU, memory, and I/O devices. It provides an interface between the hardware and the software. The kernel handles system calls, process management, memory management, device management, and file system management.")
pdf.ln(10)
pdf.multi_cell(0, 10, txt="The UNIX shell is a command-line interpreter that provides a user interface to the UNIX operating system. It interprets commands entered by the user and executes them. The most common shells are the Bourne shell (sh), the C shell (csh), and the Korn shell (ksh).")

# Save the PDF
pdf.output("test_document.pdf")

print("PDF created successfully!")
