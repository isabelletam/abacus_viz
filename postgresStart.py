import psycopg2

conn = None
try:
        conn = psycopg2.connect(database="postgres",
                                user='postgres', password='password', 
                                host='gphost08', port='5432'
        )
        
        conn.autocommit = True
        cursor = conn.cursor()

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

        with open('/System/Volumes/Data/projects/steiflab/scratch/itam/itam/A95621B.copy_number.csv', 'r') as f:
            next(f)  # Skip the header row.
            cursor.copy_from(f, 'details', sep=',')

        conn.commit()

        with open('/System/Volumes/Data/projects/steiflab/scratch/itam/itam/A95621B.gp_fit.csv', 'r') as f:
            next(f) # Skip the header row.
            cursor.copy_from(f, 'gpfit', null='', sep=',')

        conn.commit()

        # Add the auto-incrementing ID column to the 'details' table
        cursor.execute("ALTER TABLE details ADD COLUMN id SERIAL PRIMARY KEY")

        # Execute a SELECT query to retrieve all rows from the table
        cursor.execute("SELECT * FROM details")
        cursor.execute("SELECT * FROM gpfit")


except (Exception, psycopg2.DatabaseError) as error:
        print(error)
finally:
        if conn is not None:
            cursor.close()
            conn.close()
            print('Database connection closed.')
