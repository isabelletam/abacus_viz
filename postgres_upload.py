import tkinter as tk
from tkinter import filedialog
import psycopg2
import subprocess

# Global variables to store connection details
database = ""
user = ""
password = ""
host = ""
port = None
conn = None
copy_number_entry = None
gp_fit_entry = None

def submit_connection_details(database_entry, user_entry, password_entry, host_entry, port_entry):
    global database, user, password, host, port
    database = database_entry.get()
    user = user_entry.get()
    password = password_entry.get()
    host = host_entry.get()
    port = port_entry.get()
    # print("submit_connection_detailsPORT:", port)
    # print("submit_connection_detailsuser:", user)
    # print("submit_connection_detailspassword:", password)
    # print("submit_connection_detailshost:", host)
    # print("submit_connection_detailsdatabase:", database)

    return database, user, password, host, port

def upload_files(database_entry, user_entry, password_entry, host_entry, port_entry, copy_number_entry, gp_fit_entry):
    try:
        
        global database, user, password, host, port, conn
        submit_connection_details(database_entry, user_entry, password_entry, host_entry, port_entry)
        conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
        conn.autocommit = True
        cursor = conn.cursor()

        # Read file paths from the entry fields
        copy_number_path = copy_number_entry.get()
        gp_fit_path = gp_fit_entry.get()

        cmd = '''DROP TABLE IF EXISTS DETAILS '''
        cursor.execute(cmd)

        gp_cmd = '''DROP TABLE IF EXISTS GPFIT'''
        cursor.execute(gp_cmd)

        sql = '''CREATE TABLE details (
            cell_id VARCHAR(255),
            chrom VARCHAR(255),
            start INTEGER,
            end_pos INTEGER,
            two INTEGER,
            total INTEGER,
            mappability FLOAT,
            percent_gc FLOAT,
            modal_quantile FLOAT,
            modal_curve FLOAT,
            modal_corrected FLOAT,
            valid BOOLEAN,
            copy_number FLOAT,
            assignment INTEGER
        );'''

        cursor.execute(sql)

        sql = '''CREATE TABLE gpfit (
            cell_id VARCHAR(255),
            training_cell_id VARCHAR(255),
            ref_condition VARCHAR(255),
            modal_ploidy INTEGER,
            state INTEGER,
            num_bins INTEGER,
            two_coverage INTEGER,
            total_coverage FLOAT,
            predict_mean FLOAT,
            predict_std FLOAT,
            assignment FLOAT
        );'''

        cursor.execute(sql) 

        # Upload files to the corresponding tables
        with open(copy_number_path, 'r') as f:
            next(f)  # Skip header row
            cursor.copy_from(f, 'details', sep=',')
        
        with open(gp_fit_path, 'r') as f:
            next(f)  # Skip header row
            cursor.copy_from(f, 'gpfit', null='', sep=',')   
        
        # Add the auto-incrementing ID column to the 'details' table
        cursor.execute("ALTER TABLE details ADD COLUMN id SERIAL PRIMARY KEY")

        # Execute a SELECT query to retrieve all rows from the table
        cursor.execute("SELECT * FROM details")
        cursor.execute("SELECT * FROM gpfit")

        root.destroy()  # Close the Tkinter GUI window
        # # Launch Dash app if database connection was successful
        # launch_dash_app()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            cursor.close()
            conn.close()
            print('Database connection closed.')

# def launch_dash_app():
#     try:
#         dash_app_command = "python app.py"
#         print("LAUNCHING DASH APP")
#         subprocess.Popen(dash_app_command, shell=True)
#     except Exception as error:
#         print("Error launching Dash app:", error)


def open_copy_number_file():
    global copy_number_entry
    file_path = filedialog.askopenfilename()
    copy_number_entry.delete(0, tk.END)
    copy_number_entry.insert(0, file_path)

def open_gp_fit_file():
    global gp_fit_entry
    file_path = filedialog.askopenfilename()
    gp_fit_entry.delete(0, tk.END)  
    gp_fit_entry.insert(0, file_path)

def run_upload():
    global copy_number_entry, gp_fit_entry, database, user, password, host, port, root
    # Create GUI
    root = tk.Tk()
    root.title("Abacus Visualizer Setup")

    # Connection Details
    conn_label = tk.Label(root, text="Connection Details", font=("Helvetica", 16, "bold"))
    conn_label.pack()

    database_label = tk.Label(root, text="Database Name:")
    database_label.pack()

    database_entry = tk.Entry(root)
    database_entry.pack()

    user_label = tk.Label(root, text="Username:")
    user_label.pack()

    user_entry = tk.Entry(root)
    user_entry.pack()

    password_label = tk.Label(root, text="Password:")
    password_label.pack()

    password_entry = tk.Entry(root, show="*")
    password_entry.pack()

    host_label = tk.Label(root, text="Host Name:")
    host_label.pack()

    host_entry = tk.Entry(root)
    host_entry.pack()

    port_label = tk.Label(root, text="Port:")
    port_label.pack()

    port_entry = tk.Entry(root)
    port_entry.pack()

    # Copy Number File
    copy_number_label = tk.Label(root, text="Copy Number File:")
    copy_number_label.pack()

    copy_number_entry = tk.Entry(root)
    copy_number_entry.pack()

    copy_number_button = tk.Button(root, text="Open File", command=open_copy_number_file)
    copy_number_button.pack()

    # GP Fit File
    gp_fit_label = tk.Label(root, text="GP Fit File:")
    gp_fit_label.pack()

    gp_fit_entry = tk.Entry(root)
    gp_fit_entry.pack()

    gp_fit_button = tk.Button(root, text="Open File", command=open_gp_fit_file)
    gp_fit_button.pack()

    # Upload Button
    upload_button = tk.Button(root, text="Upload Files", command=lambda:upload_files(database_entry, user_entry, password_entry, host_entry, port_entry, copy_number_entry, gp_fit_entry))
    upload_button.pack()

    root.mainloop()

def get_connection_details():
    return database, user, password, host, port


run_upload()