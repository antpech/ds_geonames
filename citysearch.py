import os.path
import pandas as pd
import numpy as np
import math

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.dialects import postgresql

import pickle

from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance
from translate import Translator

from sentence_transformers import SentenceTransformer, util


class citysearch():
    engine = None
    db_params = {}
    dir_path = ''
    country_filter = []
    fclass_filter = []
    lang_filter = []
    population_filter = 0
    model_labse2 = None
    data_labse2 = pd.DataFrame()
    embeddings_labse2 = [[]]
    
    def __init__(self, dir_path='', db_params={}):
        self.db_params = db_params
        self.dir_path = dir_path

        self.country_filter = ['RU', 'AM', 'AZ', 'BY', 'GE', 'KG', 'KZ', 'MD', 'TJ', 'UA', 'UZ']
        self.fclass_filter = ['P']
        self.lang_filter = ['ru', 'az', 'en', 'tr', 'uz', 'abbr', 'iata', 'icao', 'faac']
        self.population_filter = 5000

        if self.check_dir_path() & self.check_db_params():
            try:
                self.engine = create_engine(self.db_create_connection_str())
            except:
                raise
            return
    
    def set_dir_path(self, dir_path):
        self.dir_path = dir_path
        return
        
    def set_params(self, db_params):
        self.db_params = db_params
        return
    
    def check_dir_path(self):
        return True
    
    def check_files(self):
        return True
    
    def check_db_params(self):
        return True
    
    def db_create_connection_str(self):
        conn_str = URL.create(
            drivername=self.db_params['drivername'],
            username=self.db_params['username'],
            password=self.db_params['password'],
            host=self.db_params['host'],
            port=self.db_params['port'],
            database=self.db_params['database'],
        )
        return conn_str
        
    def db_init(self, db_params={}):
        if len(db_params) != 0:
            self.db_params = db_params
            
        if not self.check_dir_path():
            return False
        
        if not self.check_db_params():
            return False

        if self.engine == None:     
            try:
                self.engine = create_engine(self.db_create_connection_str())
            except:
                raise
       
        if self.engine == None:
            return False
                
        
        query = """
        DROP TABLE IF EXISTS alternate_name CASCADE;
        DROP TABLE IF EXISTS countryinfo CASCADE;
        DROP TABLE IF EXISTS admin1codes CASCADE;
        DROP TABLE IF EXISTS geoname CASCADE;
        
        CREATE TABLE geoname (
          geoname_id INTEGER NOT NULL,
          name varchar(200) DEFAULT NULL,
          ascii_name varchar(200) DEFAULT NULL,
          alternate_names TEXT DEFAULT NULL,
          latitude NUMERIC(12,8) DEFAULT NULL,
          longitude NUMERIC(12,8) DEFAULT NULL,
          fclass varchar(1) DEFAULT NULL,
          fcode varchar(10) DEFAULT NULL,
          country varchar(2) DEFAULT NULL,
          cc2 varchar(200) DEFAULT NULL,
          admin1 varchar(20) DEFAULT NULL,
          admin2 varchar(80) DEFAULT NULL,
          admin3 varchar(20) DEFAULT NULL,
          admin4 varchar(20) DEFAULT NULL,
          population INTEGER DEFAULT NULL,
          elevation INTEGER DEFAULT NULL,
          gtopo30 INTEGER DEFAULT NULL,
          time_zone varchar(40) DEFAULT NULL,
          mod_date date DEFAULT NULL
        );

        ALTER TABLE geoname ADD CONSTRAINT PK_geoname PRIMARY KEY (geoname_id);
        CREATE INDEX IXFK_geoname_name ON geoname (name ASC);
        CREATE INDEX IXFK_geoname_ascii_name ON geoname (ascii_name ASC);
        CREATE INDEX IXFK_geoname_latitude ON geoname (latitude ASC);
        CREATE INDEX IXFK_geoname_longitude ON geoname (longitude ASC);
        CREATE INDEX IXFK_geoname_fclass ON geoname (fclass ASC);
        CREATE INDEX IXFK_geoname_fcode ON geoname (fcode ASC);
        CREATE INDEX IXFK_geoname_country ON geoname (country ASC);
        CREATE INDEX IXFK_geoname_cc2 ON geoname (cc2 ASC);
        CREATE INDEX IXFK_geoname_admin1 ON geoname (admin1 ASC);
        CREATE INDEX IXFK_geoname_population ON geoname (population ASC);
        CREATE INDEX IXFK_geoname_elevation ON geoname (elevation ASC);
        CREATE INDEX IXFK_geoname_time_zone ON geoname (time_zone ASC);

        CREATE TABLE alternate_name (
          alternate_name_id INTEGER NOT NULL,
          geoname_id INTEGER DEFAULT NULL,
          iso_language varchar(7) DEFAULT NULL,
          alternate_name varchar(200) DEFAULT NULL
        );

        ALTER TABLE alternate_name ADD CONSTRAINT PK_alternate_name PRIMARY KEY (alternate_name_id);
        CREATE INDEX IXFK_alternate_name_geoname_id ON alternate_name (geoname_id ASC);
        CREATE INDEX IXFK_alternate_name_iso_language ON alternate_name (iso_language ASC);
        CREATE INDEX IXFK_alternate_name_alternate_name ON alternate_name (alternate_name ASC);

        CREATE TABLE countryinfo (
            iso_alpha2 char(2) DEFAULT NULL,
            iso_alpha3 char(3) DEFAULT NULL,
            iso_numeric INTEGER DEFAULT NULL,
            fips_code varchar(3) DEFAULT NULL,
            name varchar(200) DEFAULT NULL,
            capital varchar(200) DEFAULT NULL,
            areainsqkm double precision,
            population INTEGER DEFAULT NULL,
            continent varchar(2) DEFAULT NULL,
            tld varchar(10) DEFAULT NULL,
            currencycode varchar(3) DEFAULT NULL,
            currencyname varchar(20) DEFAULT NULL,
            phone varchar(20) DEFAULT NULL,
            postalcode varchar(100) DEFAULT NULL,
            postalcoderegex varchar(200) DEFAULT NULL,
            languages varchar(200) DEFAULT NULL,
            geoname_id INTEGER DEFAULT NULL,
            neighbors varchar(50) DEFAULT NULL,
            equivfipscode varchar(3) DEFAULT NULL
        );

        CREATE INDEX IXFK_countryinfo_iso_alpha2 ON countryinfo (iso_alpha2 ASC);


        CREATE TABLE admin1codes (
            code varchar(11),
            name varchar(200),
            ascii_name varchar(200),
            geoname_id int
        );
        ALTER TABLE ONLY admin1codes ADD CONSTRAINT pk_admin1id PRIMARY KEY (geoname_id);
        """
        
        
        self.engine.execute(query)
        
        return True
    
    def chunker(self, seq, size):
    # from http://stackoverflow.com/a/434328
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    def db_insert_with_progress(self, df, table_name):
        chunksize = int(len(df) / 10) # 10%
        with tqdm(total=len(df)) as pbar:
            for i, cdf in enumerate(self.chunker(df, chunksize)):
                cdf.to_sql(con=self.engine, name=table_name, if_exists='append', index=False)
                pbar.update(chunksize)
        
    def db_load_data(self, dir_path='', db_params={}):

        if (dir_path.strip() != ''):
            self.dir_path = dir_path
            
        if (len(db_params) != 0):
            self.db_params = db_params
            
        if not self.check_dir_path():
            return False
        
        if not self.check_files():
            return False
        
        if not self.check_db_params():
            return False

        if self.engine == None:    
            try:
                self.engine = create_engine(self.db_create_connection_str())
            except:
                raise
       
        if self.engine == None:
            return False   
        
        self.db_init(db_params=self.db_params)

        cities = pd.read_csv(self.dir_path + '/allCountries.txt', sep='\t',
                         names=[
                             'geoname_id',
                             'name',
                             'ascii_name',
                             'alternate_names',
                             'latitude',
                             'longitude',
                             'fclass',
                             'fcode',
                             'country',
                             'cc2',
                             'admin1',
                             'admin2',
                             'admin3',
                             'admin4',
                             'population',
                             'elevation',
                             'gtopo30',
                             'time_zone',
                             'mod_date'
                         ],
                         dtype={
                             'geoname_id': int,
                             'name': object,
                             'ascii_name': object,
                             'alternate_names': object,
                             'latitude': float,
                             'longitude': float,
                             'fclass': object,
                             'fcode': object,
                             'country': object,
                             'cc2': object,
                             'admin1': object,
                             'admin2': object,
                             'admin3': object,
                             'admin4': object,
                             #'population': int,
                             #'elevation': int,
                             'gtopo30': int,
                             'time_zone': object
                             #'mod_date': datetime
                        },
                        #nrows=100,
                         index_col=False
                         )
                         #.dropna()

        query = "fclass == @self.fclass_filter \
        and country in @self.country_filter \
        and population >= @self.population_filter \
        "
        cities = cities.query(query)

        self.db_insert_with_progress(cities, 'geoname')

        print('Данные в таблицу geoname загружены.')       
        
        alter_names = pd.read_csv(self.dir_path + '/alternateNamesV2.txt', sep='\t',
                         names=[
                             'alternate_name_id',
                             'geoname_id',
                             'iso_language',
                             'alternate_name',
                             'isPreferredName',
                             'isShortName',
                             'isColloquial',
                             'isHistoric',
                             'from',
                             'to'
                         ],
                         usecols=[
                             'alternate_name_id',
                             'geoname_id',
                             'iso_language',
                             'alternate_name'
                         ],
                         #nrows=100,
                         index_col=False
                         )
                        #.dropna()
        geoname_id_unique = cities['geoname_id'].unique()
        query = """
        geoname_id in @geoname_id_unique and (iso_language in @self.lang_filter or iso_language.isna())
        """        
        alter_names = alter_names.query(query)
        self.db_insert_with_progress(alter_names, 'alternate_name')  

        print('Данные в таблицу alternate_name загружены.')    


        country_info = pd.read_csv(self.dir_path + '/countryInfo.txt', sep='\t',
                        comment='#', header=None,
                         names=[
                            'iso_alpha2',
                            'iso_alpha3',
                            'iso_numeric',
                            'fips_code',
                            'name',
                            'capital',
                            'areainsqkm',
                            'population',
                            'continent',
                            'tld',
                            'currencycode',
                            'currencyname',
                            'phone',
                            'postalcode',
                            'postalcoderegex',
                            'languages',
                            'geoname_id',
                            'neighbors',
                            'equivfipscode'
                         ],
                         #nrows=100,
                         index_col=False
                         )
                        #.dropna() 
        
        self.db_insert_with_progress(country_info, 'countryinfo')   

        print('Данные в таблицу countryinfo загружены.') 

        admin1codes = pd.read_csv(self.dir_path + '/admin1CodesASCII.txt', sep='\t',
                        comment='#', header=None,
                         names=[
                            'code',
                            'name',
                            'ascii_name',
                            'geoname_id'
                         ],
                         #nrows=100,
                         index_col=False
                         )
                        #.dropna() 
        
        self.db_insert_with_progress(admin1codes, 'admin1codes')

        print('Данные в таблицу admin1codes загружены.')  
                
        return True
    
    def pickle_data(self, file_name='', data=''):
        with open(file_name + '.pkl', 'wb') as fout:
            pickle.dump((data), fout)
        
        return True
    
    def unpickle_data(self, file_name=''):
        with open(file_name + '.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    
    def get_CORPUS(self, lang=['en']):
        query = """
        select g.geoname_id, cast(null as int) as alternate_name_id, g.name as "name"
        from geoname g
        
        union all
        
        select an.geoname_id
        ,an.alternate_name_id 
        ,an.alternate_name
        from alternate_name an 
        where 
        (not exists(select geoname_id  from geoname gn where gn.geoname_id = an.geoname_id and gn.name = an.alternate_name))
        """

        lang_str = ''.join("'"+str(el)+"', " for el in lang)[:-2]
        if len(lang) != 0 and lang[0]  != 'all':
            query += " and an.iso_language in ("+lang_str+")"

        CORPUS = pd.read_sql_query(query, con=self.engine).dropna(subset=['name'])

        return CORPUS

    
    def calc_count_vectorizer(self):
        CORPUS = self.get_CORPUS(lang=['all'])
        CORPUS['count_vectorizer'] = ''

        bow = CountVectorizer(analyzer='char', ngram_range=(1,2))
        bow.fit(CORPUS)
        self.pickle_data(file_name='count_vectorizer', data=bow)

        arr_vector = []
        for name in CORPUS['name']:
            word_vector = bow.transform([name]).toarray()[0]
            arr_vector.append(word_vector.tolist())

        CORPUS['count_vectorizer'] = arr_vector
        
        CORPUS[['alternate_name_id', 'geoname_id', 'count_vectorizer']].to_sql(
            'CountVectorizer', 
            self.engine, 
            if_exists="replace",
            dtype={
                'geoname_id': sqlalchemy.types.INTEGER,
                'alternate_name_id': sqlalchemy.types.INTEGER, 
                'count_vectorizer': postgresql.ARRAY(sqlalchemy.types.INTEGER)
            }
        )

        query = """
            delete from public."CountVectorizer" cv
            where count_vectorizer = '{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}'
        """
        self.engine.execute(query)

        return
        
    def get_CV_similarity(self, search, translate=False, lang='ru'):
        translated_search = search
        if translate:
            translator = Translator(from_lang=lang, to_lang='en')
            translated_search = translator.translate(search)
        
        search_len = (len(translated_search)-3)

        query = """
            SELECT cv.geoname_id
            , cv.count_vectorizer
            , coalesce(an.alternate_name, g.ascii_name) as name_en
            , g."name"
            , c."name" as country
            , ad."name" as region
            , g.population 
            FROM public."CountVectorizer" cv
            inner join geoname g on cv.geoname_id = g.geoname_id
            inner join countryinfo c on g.country = c.iso_alpha2 
            inner join admin1codes ad on CONCAT (g.country, '.', cast(g.admin1 as varchar(11)))  = ad.code 
            left outer join alternate_name an ON cv.alternate_name_id = an.alternate_name_id
            WHERE length(coalesce(an.alternate_name, g.ascii_name)) >= """ + str(search_len)
        CORPUS = pd.read_sql_query(query, con=self.engine)
        CORPUS['cosine_similarity'] = 0
            
        bow = citysearch.unpickle_data(self, file_name='count_vectorizer')
        search_vector = bow.transform([translated_search]).toarray()[0]

        arr_sim = []
        for word_vector in CORPUS['count_vectorizer']:    
            word_vector = np.array(word_vector)
            
            cosine_similarity = round(1 - distance.cosine(word_vector, search_vector), 3)
            arr_sim.append(cosine_similarity)
            
        CORPUS['cosine_similarity'] = arr_sim

        ret_df = CORPUS[['geoname_id', 'name', 'country', 'region', 'population', 'cosine_similarity']] \
        .sort_values(by=['cosine_similarity', 'population'], ascending=False) \
        .head(5)

        ret_df = ret_df.drop_duplicates(subset=['geoname_id', 'name', 'country', 'region', 'population'], keep='first')

        return ret_df.to_dict('records')
    

    def calc_sbertv2(self):
        CORPUS = self.get_CORPUS(lang=['all'])

        query = """DROP TABLE IF EXISTS public.sbert_multiv2 CASCADE;"""
        self.engine.execute(query)

        model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

        CORPUS['embeddings'] = pd.Series(dtype='object')

        corpus_len = CORPUS.shape[0]
        batch_size = 512
        if batch_size > corpus_len:
            batch_size = corpus_len
        batch_num = math.ceil(corpus_len / batch_size)

        for i in range(batch_num):
            start = i * batch_size
            end = start + batch_size

            df = CORPUS[start:end].copy().reset_index()
            

            embeddings = model.encode(df['name'], 
                                    normalize_embeddings=True, 
                                    show_progress_bar=False)
            embeddings = np.array(embeddings).tolist()

            df['embeddings'] = embeddings
            
            df[['alternate_name_id', 'geoname_id', 'embeddings']].to_sql(
                'sbertv2', 
                self.engine,
                index=False, 
                if_exists="append",
                dtype={
                    'geoname_id': sqlalchemy.types.INTEGER, 
                    'embeddings': postgresql.ARRAY(sqlalchemy.types.FLOAT)
                }
            )

        return
    
    def get_sbertv2_similarity(self, search):     
        search_len = (len(search)-3)

        query = """
            SELECT sb.geoname_id
            , sb.embeddings
            , coalesce(an.alternate_name, g.ascii_name) as name_en
            , g."name"
            , c."name" as country
            , ad."name" as region
            , g.population 
            FROM public.sbertv2 sb
            inner join geoname g on sb.geoname_id = g.geoname_id
            inner join countryinfo c on g.country = c.iso_alpha2 
            inner join admin1codes ad on CONCAT (g.country, '.', cast(g.admin1 as varchar(11)))  = ad.code 
            left outer join alternate_name an ON sb.alternate_name_id = an.alternate_name_id
            WHERE length(coalesce(an.alternate_name, g.ascii_name)) >= """ + str(search_len)
        CORPUS = pd.read_sql_query(query, 
                                   con=self.engine
                                   )
        embeddings = CORPUS['embeddings'].transform(np.array).transform(pd.Series).values.astype('float32')
           
        model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        search_embedding = model.encode(search, normalize_embeddings=True, show_progress_bar=False)

        res = util.semantic_search(search_embedding, embeddings, top_k=100)

        idx = [i['corpus_id'] for i in res[0]]
        score = [i['score'] for i in res[0]]

        ret_df = CORPUS.loc[idx, ['geoname_id', 'name', 'country', 'region', 'population']] \
                    .assign(similarity=score) \
                    .drop_duplicates(subset=['geoname_id', 'name', 'country', 'region', 'population']) \
                    .iloc[:5]

        return ret_df.to_dict('records')
    
    def calc_sbert_Labse(self):
        CORPUS = self.get_CORPUS(lang=['all'])

        query = """DROP TABLE IF EXISTS public.sbert_Labse CASCADE;"""
        self.engine.execute(query)

        model = SentenceTransformer('dima-does-code/LaBSE-geonames-15K-MBML-5e-v1')

        CORPUS['embeddings'] = pd.Series(dtype='object')

        corpus_len = CORPUS.shape[0]
        batch_size = 512
        if batch_size > corpus_len:
            batch_size = corpus_len
        batch_num = math.ceil(corpus_len / batch_size)

        for i in range(batch_num):
            start = i * batch_size
            end = start + batch_size

            df = CORPUS[start:end].copy().reset_index()
            

            embeddings = model.encode(df['name'], 
                                    normalize_embeddings=True, 
                                    show_progress_bar=False)
            embeddings = np.array(embeddings).tolist()

            df['embeddings'] = embeddings
            
            df[['alternate_name_id', 'geoname_id', 'embeddings']].to_sql(
                'sbert_Labse', 
                self.engine,
                index=False, 
                if_exists="append",
                dtype={
                    'geoname_id': sqlalchemy.types.INTEGER, 
                    'embeddings': postgresql.ARRAY(sqlalchemy.types.FLOAT)
                }
            )

        return
    
    def get_sbert_Labse_similarity(self, search):     
        search_len = (len(search)-3)

        
        query = """
            SELECT sb.geoname_id
            , sb.embeddings
            , coalesce(an.alternate_name, g.ascii_name) as name_en
            , g."name"
            , c."name" as country
            , ad."name" as region
            , g.population 
            FROM public."sbert_Labse" sb
            inner join geoname g on sb.geoname_id = g.geoname_id
            inner join countryinfo c on g.country = c.iso_alpha2 
            inner join admin1codes ad on CONCAT (g.country, '.', cast(g.admin1 as varchar(11)))  = ad.code 
            left outer join alternate_name an ON sb.alternate_name_id = an.alternate_name_id
            WHERE length(coalesce(an.alternate_name, g.ascii_name)) >= """ + str(search_len)

        CORPUS = pd.read_sql_query(query, 
                                   con=self.engine
                                   )
        embeddings = CORPUS['embeddings'].transform(np.array).transform(pd.Series).values.astype('float32')
        # embeddings =  np.load('embeddings_labse.npy')
           
        model = SentenceTransformer('dima-does-code/LaBSE-geonames-15K-MBML-5e-v1')
        search_embedding = model.encode(search, normalize_embeddings=True, show_progress_bar=False)

        res = util.semantic_search(search_embedding, embeddings, top_k=100)

        idx = [i['corpus_id'] for i in res[0]]
        score = [i['score'] for i in res[0]]

        ret_df = CORPUS.loc[idx, ['geoname_id', 'name', 'country', 'region', 'population']] \
                    .assign(similarity=score) \
                    .drop_duplicates(subset=['geoname_id', 'name', 'country', 'region', 'population']) \
                    .iloc[:5]

        return ret_df.to_dict('records')
    


    def calc_sbert_Labse2(self):
        CORPUS = self.get_CORPUS(lang=['all'])

        if self.model_labse2 == None:   
            self.model_labse2 = SentenceTransformer('dima-does-code/LaBSE-geonames-15K-MBML-5e-v1')
        model = self.model_labse2

        CORPUS['embeddings'] = pd.Series(dtype='object')

        corpus_len = CORPUS.shape[0]
        batch_size = 512
        if batch_size > corpus_len:
            batch_size = corpus_len
        batch_num = math.ceil(corpus_len / batch_size)

        all_embeddings = np.zeros([1,768])

        for i in range(batch_num):
            start = i * batch_size
            end = start + batch_size

            df = CORPUS[start:end].copy().reset_index()
            

            embeddings = model.encode(df['name'], 
                                    normalize_embeddings=True, 
                                    show_progress_bar=False)
            embeddings = np.array(embeddings)
            all_embeddings = np.concatenate((all_embeddings, embeddings), axis=0)

        all_embeddings = np.delete(all_embeddings, (0), axis=0)        
        
        # Пикл может сохранить в любом виде, а вот нормально распаковывает только в таком:
        CORPUS['embeddings'] = np.array(all_embeddings).tolist()
        self.pickle_data(file_name='sbert_Labse2', data=CORPUS)

        #Долгая операция.... Надо изменить
        self.embeddings_labse2 = CORPUS['embeddings'].transform(np.array).transform(pd.Series).values.astype('float32')
        self.data_labse2 = CORPUS

        return
    
    def get_sbert_Labse_similarity2(self, search): 
        #Повторно используем ранее загруженные данные            
        if self.data_labse2.empty:
            self.data_labse2 = self.unpickle_data(file_name='sbert_Labse2')            
            self.embeddings_labse2 = self.data_labse2['embeddings'].transform(np.array).transform(pd.Series).values.astype('float32')

        CORPUS = self.data_labse2
        embeddings = self.embeddings_labse2
        
        #Повторно используем ранее загруженную модель
        if self.model_labse2 == None:   
            self.model_labse2 = SentenceTransformer('dima-does-code/LaBSE-geonames-15K-MBML-5e-v1')
        model = self.model_labse2

        search_embedding = model.encode(search, normalize_embeddings=True, show_progress_bar=False)

        res = util.semantic_search(search_embedding, embeddings, top_k=100)

        idx = [i['corpus_id'] for i in res[0]]
        score = [i['score'] for i in res[0]]

        top_df = CORPUS.loc[idx, ['geoname_id']] \
                    .assign(similarity=score) \
                    .drop_duplicates(subset=['geoname_id']) \
                    .iloc[:5]
        ids = ','.join(str(id) for id in top_df['geoname_id'])

        query = """
            SELECT g.geoname_id
            , g."name"
            , c."name" as country
            , ad."name" as region
            , g.population 
            FROM geoname g
            inner join countryinfo c on g.country = c.iso_alpha2 
            inner join admin1codes ad on CONCAT (g.country, '.', cast(g.admin1 as varchar(11)))  = ad.code 
            WHERE g.geoname_id in (""" + ids + """)
        """
        ret_df = pd.read_sql_query(query, 
                                   con=self.engine
                                   )
        
        ret_df = ret_df.merge(top_df, on='geoname_id')
        ret_df = ret_df.query('similarity >= 0.75').sort_values(by=['similarity', 'population'], ascending=False)

        return ret_df.to_dict('records')
