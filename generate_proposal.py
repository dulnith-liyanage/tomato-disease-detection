from fpdf import FPDF

class ProposalPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'Project Proposal', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 8, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Helvetica', '', 11)
        self.multi_cell(0, 6, body)
        self.ln(6)

def create_pdf():
    pdf = ProposalPDF()
    pdf.add_page()

    pdf.chapter_title('Project Title')
    pdf.chapter_body('EcoSort: An AI-Powered Automated Smart Waste Sorting System')

    pdf.chapter_title('Problem Statement')
    problem = (
        "Proper waste segregation is critical for environmental sustainability, as contaminated recycling streams "
        "cost municipalities millions of dollars annually and result in tons of recyclable materials being diverted to landfills. "
        "In public spaces, humans frequently misclassify their waste due to confusing guidelines or negligence, leading to high "
        "contamination rates. An automated system that removes human error from the sorting process is essential to improving "
        "recycling efficiency, reducing landfill footprint, and advancing smart city infrastructure."
    )
    pdf.chapter_body(problem)

    pdf.chapter_title('Proposed Solution & Hardware')
    solution = (
        "We propose building an automated 'Smart Bin' equipped with an Edge AI computer vision pipeline to classify and sort waste "
        "in real-time. \n\n"
        "Platform & Hardware:\n"
        "- Core Processor: Raspberry Pi 4 (or equivalent microcontroller for edge computing).\n"
        "- Vision: Raspberry Pi Camera Module to capture images of the discarded item.\n"
        "- Actuators: Servo motors to physically tilt a sorting platform or open the correct internal chute (Trash, Recycling, Compost).\n"
        "- AI Model: A highly accurate, quantized Convolutional Neural Network (such as MobileNetV2 or YOLO) deployed via TensorFlow Lite.\n\n"
        "Implementation:\n"
        "When an item is placed on the bin's scanning tray, an ultrasonic sensor triggers the camera. The Raspberry Pi processes the image "
        "using the TFLite model, classifying the item into one of the designated categories. Based on the model's high-confidence prediction, "
        "a GPIO signal is sent to the servo motors to drop the item into the correct bin compartment, completely automating the segregation process."
    )
    pdf.chapter_body(solution)

    pdf.chapter_title('Objectives')
    objectives = (
        "By the end of the project term, we aim to achieve the following:\n"
        "1. Hardware Assembly: Construct a functional prototype of a smart bin with an integrated camera, scanning tray, and servo-driven sorting mechanism.\n"
        "2. Model Training: Train and deploy a lightweight, quantized image classification model capable of distinguishing between common waste categories (e.g., plastic bottles, aluminum cans, paper, and general trash) with an accuracy exceeding 90%.\n"
        "3. System Integration: Successfully integrate the AI software pipeline with the physical hardware, achieving real-time inference and mechanical sorting within 2 seconds per item.\n"
        "4. Demonstration: Provide a live demonstration of the system correctly sorting a mixed set of items, validating its potential for real-world environmental impact."
    )
    pdf.chapter_body(objectives)

    pdf.output('Project_Proposal_EcoSort.pdf')
    print("PDF generated successfully.")

if __name__ == '__main__':
    create_pdf()
