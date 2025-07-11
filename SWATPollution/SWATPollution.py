import os
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error
import plotly.express as px
from bokeh.models import HoverTool
import geoviews as gv
gv.extension('bokeh', 'matplotlib')
import plotly.graph_objects as go
from .pollution_utils import observacions_from_conca, generate_pollution_observations
from pySWATPlus.TxtinoutReader import TxtinoutReader
from pathlib import Path
import importlib.resources
from . import rivs1

def nse(observations, predictions):
    return 1 - (sum((observations - predictions)**2) / sum((observations - observations.mean())**2))

def pbias(observations, predictions):
    return 100 * sum(observations - predictions) / sum(observations)


# Root Mean Standard Deviation Ratio (RSR)
def rsr(observations, predictions):
    #return root_mean_squared_error(observations, predictions) / (sum((observations - predictions.mean())**2))**0.5
    return root_mean_squared_error(observations, predictions) / np.std(observations)
    

class SWATPollution:
    """
    Class to manage the setup and optional execution of a SWAT simulation 
    for contaminant transport in a specified watershed.

    Parameters
    ----------
    conca : str
        Name of the watershed to be modeled. Valid options include 'muga', 'fluvia', 'ter', 
        'llobregat', 'besos', 'tordera', 'sud'.

    contaminant : str
        Name of the pollutant to simulate (e.g., 'Venlafaxina', 'Ciprofloxacina').

    txtinout_folder : str or Path
        Path to the directory containing the SWAT txtinout files.

    channels_geom_path : str or Path, optional
        Path to the shapefile representing the channel geometry. If None, a default file is used.

    tmp_path : str or Path, optional
        Temporary directory where txtinout files will be copied before execution (only if `run=True`).

    run : bool, optional
        If True, prepares and runs the SWAT model. Defaults to False.

    compound_features : dict, optional
        Dictionary specifying SWAT file modifications. 
        Format: {'filename': (id_col, [(id, col, value)])}.

    show_output : bool, optional
        If True, shows the output logs of the SWAT run. Defaults to True.

    copy_txtinout : bool, optional
        If True, copies txtinout files to `tmp_path` before execution (only if `run=True`).

    overwrite_txtinout : bool, optional
        If True, allows overwriting existing files in `tmp_path` (only if `run=True`).

    observacions : pd.DataFrame, optional
        Observational data for calibration and evaluation. If None, loads default observations.

    lod_path : str or Path, optional
        Path to the file containing limits of detection (LOD). If None, uses the default.

    year_start : int, optional
        Start year of the simulation. Defaults to 2000. Used only if `run=True`.

    year_end : int, optional
        End year of the simulation. Defaults to 2022. Used only if `run=True`.

    warmup : int, optional
        Number of warm-up years. Defaults to 1. Used only if `run=True`.

    Example
    -------
    >>> from pathlib import Path
    >>> import pandas as pd
    >>> contaminant = 'Venlafaxina'
    >>> conca = 'besos'
    >>> cwd = Path('C:/Users/joans/OneDrive/Escriptori/icra/traca_contaminacio/traca_contaminacio')
    >>> txtinout_folder = cwd / 'data' / 'txtinouts' / 'tmp' / contaminant / conca
    >>> channels_geom_path = cwd / 'data' / 'rivs1' / 'canals_tot_ci.shp'
    >>> recall_points_path = cwd.parent / 'traca' / 'traca' / 'inputs compound generator' / 'inputs' / 'recall_points.xlsx'
    >>> recall_points_df = pd.read_excel(recall_points_path)
    >>> edars = recall_points_df[recall_points_df['conca'] == conca][['lat', 'lon', 'edar_code']].dropna()
    >>> observacions = generate_pollution_observations(contaminant)
    >>> df = observacions_from_conca(channels_geom_path, observacions, conca)
    >>> first_observation = df.year.min()
    >>> year_end = 2022
    >>> year_start = max(first_observation - 3, 2000)
    >>> warmup = max(1, first_observation - year_start)
    >>> swatpy = SWATPollution(
    ...     conca=conca,
    ...     contaminant=contaminant,
    ...     txtinout_folder=txtinout_folder,
    ...     channels_geom_path=channels_geom_path,
    ...     run=False,
    ...     year_start=year_start,
    ...     year_end=year_end,
    ...     warmup=warmup,
    ...     observacions=observacions
    ... )
    """
    
    

    def __init__(self, 
                 conca: str, 
                 contaminant: str, 
                 txtinout_folder: str | Path, 
                 channels_geom_path: str | Path | None = None,
                 tmp_path: str | Path | None = None,
                 run: bool = False,
                 compound_features: dict | None = None,    #{filename: (id_col, [(id, col, value)])}
                 show_output: bool = True,
                 copy_txtinout: bool = True,
                 overwrite_txtinout: bool = False,
                 observacions: pd.DataFrame | None = None,
                 lod_path: str | Path | None = None,
                 year_start: int | None = 2000,
                 year_end: int | None = 2022,
                 warmup: int | None = 1

    ):
        
        if not tmp_path:
            tmp_path = os.path.join('data', 'txtinouts', 'tmp')
        
        
        if observacions is None:
            observacions = generate_pollution_observations(contaminant)

        if channels_geom_path is None:
            # Access the 'inputs' directory inside the installed package
            with importlib.resources.path(rivs1, "canals_tot_ci.shp") as shp_path:
                gdf = gpd.read_file(shp_path)
                observacions_conca = observacions_from_conca(shp_path, observacions, conca)
        else:
            gdf = gpd.read_file(channels_geom_path)
            observacions_conca = observacions_from_conca(channels_geom_path, observacions, conca)

                
        if compound_features is None:
            compound_features = {}
            
        if lod_path is None:
            with importlib.resources.path("SWATPollution", "") as lod_path:
                lod_path = lod_path / 'lod.xlsx'
                lod_df = pd.read_excel(lod_path, index_col=0)
        else:
            lod_df = pd.read_excel(lod_path, index_col=0)
        self.lod = lod_df.loc[contaminant, 'LOD (mg/L)']
            
        pollutant_name_joined = contaminant

        if run:
            reader = TxtinoutReader(txtinout_folder)

            if copy_txtinout:
                tmp_path = reader.copy_swat(tmp_path, overwrite = overwrite_txtinout)
                reader = TxtinoutReader(tmp_path)

            #set up
            reader.set_beginning_and_end_year(year_start, year_end)
            reader.set_warmup(warmup)
            reader.enable_object_in_print_prt('channel_sd', True, False, False, False)
            reader.enable_object_in_print_prt('poll', True, False, False, False)
            reader.disable_csv_print()


            #delete pollutants that are not the current one
            file = reader.register_file('pollutants.def', has_units = False, index = 'name')
            df = file.df
            df = df[df['name'] == pollutant_name_joined]
            file.df = df.copy()
            file.overwrite_file()

            file = reader.register_file('pollutants_om.exc', has_units = False)
            df = file.df
            df = df[df['pollutants_pth'] == pollutant_name_joined]
            file.df = df.copy()
            file.overwrite_file()
        
            #guardar abans parametres a escriure
            txt_in_out_result = reader.run_swat(compound_features, show_output=show_output)

        else:
            txt_in_out_result = txtinout_folder


        reader = TxtinoutReader(txt_in_out_result)
        self.reader = reader
        self.txtinout_path = txt_in_out_result

        pollutant_namde_joined = contaminant
        
        poll = reader.register_file('channel_poll_day.txt', 
                            has_units=False,
                            usecols=['mon', 'day', 'yr', 'gis_id', 'pollutant', 'sol_out_mg', 'sor_out_mg'],
                            filter_by={
                                'pollutant': pollutant_namde_joined
                            })
                
        poll_df = poll.df
        poll_df['tot_out_mg'] = poll_df['sol_out_mg'] + poll_df['sor_out_mg'] #mg per day
        poll_df['tot_out_mg_sec'] = poll_df['tot_out_mg'] / (24 * 60 * 60) #mg per second       
        
        #read flow
        flow = reader.register_file('channel_sdmorph_day.txt', 
                    has_units=True,
                    usecols=['mon', 'day', 'yr', 'gis_id', 'flo_out'])
        
        self.flow = flow.df
        flow_df = flow.df   #flo_out in m3/s
        
        #merge
        df = poll_df.merge(flow_df, left_on=['mon', 'day', 'yr', 'gis_id'], right_on=['mon', 'day', 'yr', 'gis_id'])
        df['mg_m3'] = df['tot_out_mg_sec'] / df['flo_out']
        df['mg_l'] = df['mg_m3'] / 1000
        df = df[['mon', 'day', 'yr', 'gis_id', 'pollutant', 'mg_l', 'flo_out', 'tot_out_mg']]
        
        
        df['pollutant'] = df['pollutant'].str.replace(pollutant_name_joined, contaminant)
        

        #merge observacions with model results
        merged_df = observacions_conca.merge(df, left_on=['variable', 'gis_id', 'year', 'month', 'day'], right_on=['pollutant', 'gis_id', 'yr', 'mon', 'day'])


        #calculate error
        merged_df['error'] = merged_df['valor'] - merged_df['mg_l']
        merged_df['error'] = merged_df['error'].abs()
        #merged_df['error'] = merged_df['error'] / merged_df['valor']
        #merged_df['error'] = merged_df['error'].replace([np.inf, -np.inf], np.nan).dropna()
        #merged_df['error'] = merged_df['error'] * 100

        
        aux_df = merged_df[['valor', 'mg_l']]
        aux_df = aux_df.dropna()
        observations = aux_df['valor'].values
        predictions = aux_df['mg_l'].values


        #filter channels by conca and reporject to EPSG:4326
        gdf = gdf[gdf['layer'] == conca]
        gdf = gdf.to_crs(epsg=4326)
        
        #merge results with channel geometry and convert to geodataframe
        df = df.merge(gdf[['Channel', 'geometry', 'layer']], left_on = ['gis_id'], right_on=['Channel'])
        df.rename(columns = {'mon':'month', 'yr':'year'}, inplace=True)
        df["Date"] = pd.to_datetime(df[["year", "month", "day"]])
        gdf_map = gpd.GeoDataFrame(df, geometry='geometry')
        gdf_map['date_str'] = gdf_map['Date'].dt.strftime('%Y-%m-%d')

        self.df = df


        #convert observations to geodataframe
        gdf_observacions = gpd.GeoDataFrame(
            merged_df, geometry=gpd.points_from_xy(merged_df.utm_x, merged_df.utm_y), crs="EPSG:25831"
        )


        gdf_observacions = gdf_observacions.to_crs(epsg=4326)
        gdf_observacions['fecha_str'] = gdf_observacions['fecha'].dt.strftime('%Y-%m-%d')

        df_error = pd.DataFrame({'obs':observations, 'pred':predictions})
        df_error = df_error.replace([np.inf, -np.inf], np.nan).dropna()

        #drop rows where pred is 0 (we would be wasting time trying to optimize this points, if it's 0 is because we are not generating anything)
        df_error = df_error[df_error['pred'] > 0]

        if np.isnan(self.lod):
            self.lod = None

        #if not non or nan
        if self.lod is not None:
            df_error = df_error[df_error['obs'] > self.lod]
            
        self.df_error = df_error

            
        self.error = -1 * r2_score(df_error['obs'].values, df_error['pred'].values) #negated r2 score
        #self.rmse = mean_squared_error(df_error['obs'].values, df_error['pred'].values, squared=False) * 1e6
        self.mape = mean_absolute_percentage_error(df_error['obs'].values, df_error['pred'].values)
        self.nse = nse(df_error['obs'].values, df_error['pred'].values)
        self.pbias = pbias(df_error['obs'].values, df_error['pred'].values)
        self.rsr = rsr(df_error['obs'].values, df_error['pred'].values)




        self.river_map = gdf
        self.gdf_map = gdf_map
        self.gdf_observacions = gdf_observacions
        self.contaminant = contaminant
        self.conca = conca
        self.df_error = df_error

    
    def get_df(self):
        df = self.df.copy()
        df.rename(columns={'flo_out': 'flo_out (m3/s)', 'tot_out_mg_l': 'tot_out (mg/day)', 'tot_out_mg_l': 'tot_out (mg/l)'}, inplace=True)
        return df
    
    def get(self, df_name, has_units = False):
        reader = self.reader
        file = reader.register_file(df_name, has_units=has_units)
        return file.df
    
    def get_observacions(self):
        return self.gdf_observacions

    def get_txtinout_path(self):
        return self.txtinout_path
    
    def simple_map(self, edars):
    
        # Create the 'observations_map' plot    
        hover = HoverTool(tooltips=[("observacio (mg/l)", "@valor"),
                                    ("prediccio (mg/l)", "@mg_l"),
                                    ("error (mg/l)", "@error")
                                ])


        gdf_observacions = self.gdf_observacions[['geometry', 'valor', 'mg_l', 'error']]
        gdf_observacions = gdf_observacions.replace([np.inf, -np.inf], np.nan).dropna()

        #for each point with multiple observations, take the mean for color
        error_gdf = gdf_observacions.groupby(['geometry']).agg(max_error=('error', np.max)).reset_index()
        error_gdf = error_gdf.replace([np.inf, -np.inf], np.nan).dropna()
        error_gdf = gpd.GeoDataFrame(error_gdf, geometry='geometry')

        #merge on geometry with gdf_observacions
        gdf_observacions = gdf_observacions.merge(error_gdf, left_on = ['geometry'], right_on=['geometry'])
        
        observations_map = gv.Points(
            gdf_observacions
        ).opts(tools=[hover], color='max_error', show_legend=True, cmap = 'Reds', line_color = 'black', size=7, colorbar=True)

        
        hover_2 = HoverTool(tooltips=[("canal", "@Channel"),
                                ])

        # Create the 'river_map' plot
        river_map = gv.Path(
            self.river_map,
            ).opts(tools=[hover_2])

    
        tiles = gv.tile_sources.CartoLight()  

        
        gdf_edars = gpd.GeoDataFrame(
            edars, geometry=gpd.points_from_xy(edars.lon, edars.lat), crs="EPSG:4326"
        )
        edars_map = gv.Points(
            gdf_edars
        ).opts( show_legend=True, marker = 's', color = 'blue', line_color = 'black', size=4, colorbar=True)


        
        return (tiles * river_map * observations_map * edars_map).opts(width=800, height=500)
          
    def scatter_plot(self, title = None, path = None):
        df = self.gdf_observacions.copy()

        df['ng_l'] = df['mg_l']*1e6
        df['valor_ng_l'] = df['valor']*1e6

        df = df.rename(columns = {'ng_l':'prediccio (ng/l)', 'valor_ng_l':'observacio (ng/l)'})        

        if self.lod is not None:
            df = df[df['observacio (ng/l)'] > self.lod]

        df = df[df['prediccio (ng/l)'] > 0]
        
        fig = go.Figure()
        
        fig = px.scatter(df, x="observacio (ng/l)", y="prediccio (ng/l)", hover_data=["gis_id"])

        axis_range = max( max(df["prediccio (ng/l)"]), max(df["observacio (ng/l)"]) ) + 20
                    
        fig.add_shape(
            type="line",
            x0=0, y0=0,
            x1=axis_range, y1=axis_range,
            line=dict(width=1, dash='dot', color='black'),
            name='x=y',
        )
        
        fig.update_traces(marker=dict(size=9,
                                      line=dict(width=1)),
                            selector=dict(mode='markers'))

        fig.update_layout(
            showlegend=True,
            plot_bgcolor="white",
            legend_title_text='',
            title = title,
            legend=dict(
                orientation='h',  # horizontal legend
                yanchor='bottom',  # anchor the legend to the bottom
                y=1.15,  # position the legend just above the bottom
                xanchor='right',  # anchor the legend to the right
                x=1  # position the legend at the far right
            )

        )
        
        fig.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey',
            title="Observation (ng/l)",
            range=[0, axis_range]  # Ensure x-axis starts at 0
        )

        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey',
            title="Prediction (ng/l)",
            range=[0, axis_range]  # Ensure y-axis starts at 0
        )
        
        fig.update_layout(font=dict(family="Arial"))


        if path is not None:
            fig.write_image(path, width=600, height=600)
        
        return fig
    
   
    
    def store_scatter_csv(self, path):
        df = self.gdf_observacions.copy()

        #df = df.replace([np.inf, -np.inf], np.nan).dropna()

        df['ng_l'] = df['mg_l']*1e6
        df['valor_ng_l'] = df['valor']*1e6

        df = df.rename(columns = {'ng_l':'prediccio (ng/l)', 'valor_ng_l':'observacio (ng/l)'})
      


        if self.lod is not None:
            df = df[df['observacio (ng/l)'] > self.lod]

        df = df[df['prediccio (ng/l)'] > 0]
        
        df.to_csv(path, index=False)
        
    
    def r2_mass(self):

        df = self.gdf_observacions.copy()
        df['mg'] = df['mg_l'] * df['flo_out'] * 1000
        df['valor'] = df['valor'] * df['flo_out'] * 1000


        return (-1 * r2_score(df['valor'].values, df['mg'].values))


    def get_error(self):
        return self.error
        #return self.r2_mass()
        
    def visualise_map(self, attribute, name, units, day, month, year):

        #return self.visualise_map_matplotlib('mg_l', 30, day, month, year)
        gdf_map = self.gdf_map[(self.gdf_map['Date'].dt.year == year) & (self.gdf_map['Date'].dt.month == month) & (self.gdf_map['Date'].dt.day == day)]

        if (len(gdf_map) == 0):
            raise Exception("No data for this date")

        gdf_map = gdf_map.copy()

        #apply sqrt scale
        gdf_map['color'] = gdf_map[attribute].apply(lambda x: np.sqrt(x))

        gdf_map = gdf_map.replace([np.inf, -np.inf], np.nan).dropna()
        
        river_map = gdf_map.copy()

        tiles = gv.tile_sources.CartoLight()  # You can choose from different tile sources
        
        atribute_map = gv.Path(
            gdf_map,
            ).opts(color='color', show_legend=True, cmap = 'Reds', colorbar=True, line_width=2.5)

        hover = HoverTool(tooltips=[("cabal", "@flo_out"),
                                    (f"{name} ({units})", f"@{attribute}"),
                                    ])
        river = gv.Path(
            river_map,
            ).opts(tools=[hover], color='black', line_width=3, alpha=0.5)

        return (river * atribute_map * tiles).opts(width=800, height=500)


    def visualise_load(self, day = 1, month = 1, year = 2001):
        return self.visualise_map('tot_out_mg', 'load', 'mg/day', day, month, year)
    
    def visualise_concentration(self, day = 1, month = 1, year = 2001):
        return self.visualise_map('mg_l', 'concentation', 'mg/l', day, month, year)
        
    def plot_channel(self, gis_id, title = None, path = None):

        predictions = self.gdf_map.copy()
        observations = self.gdf_observacions.copy()

        predictions['ng_l'] = predictions['mg_l']*1e6
        observations['valor_ng_l'] = observations['valor']*1e6

        predictions_channel = predictions[predictions['gis_id'] == gis_id][['ng_l', 'Date', 'flo_out']]
        observations_channel = observations[observations['gis_id'] == gis_id].copy()

        observations_channel['color'] = observations_channel['origen'].apply(lambda x: '#0d920d' if x == 'aca' else '#fe7c09')
        observations_channel['origen'] = observations_channel['origen'].apply(lambda x: 'Observations by ACA' if x == 'aca' else 'Observations by ICRA')
        observations_channel = observations_channel.rename(columns = {'valor_ng_l':'observacio (ng/l)'})
        #observations_channel = observations_channel.rename(columns = {'ng_l':'prediccio (ng/l)', 'valor_ng_l':'observacio (ng/l)'})

        fig1 = go.Figure()
        fig1.layout.font.family = 'Arial'

        #hover_data=["gis_id"], 
        """
        fig1 = px.scatter(
            observations_channel, 
            x="fecha", 
            y="observacio (ng/l)", 
            color='origen', 
            color_discrete_sequence='color', 
            color_discrete_map=dict(zip(observations_channel['origen'], 
                                        observations_channel['color'])))
        """

        
        fig1.add_trace(
            go.Scatter(x=predictions_channel['Date'], 
                    y=predictions_channel['flo_out'],
                    mode='lines', 
                    line=dict(color='#80b1d3'),
                    name='Predicted Discharge',
                    yaxis='y2',
                    opacity=0.5  # Adjust transparency (0 = fully transparent, 1 = opaque)
                    
                    )
        )

        
        fig1.add_trace(
            go.Scatter(x=predictions_channel['Date'], 
                    y=predictions_channel['ng_l'],
                    mode='lines', 
                    line=dict(color='#1f77b4'),
                    name='Predicted Concentration')
        )

        
        fig1.add_trace(go.Scatter(
            x=observations_channel['fecha'],
            y=observations_channel['observacio (ng/l)'],
            mode='markers',
            name='Observations',
            marker=dict(
                color='rgb(220, 20, 60)'  # Dark red color for all markers
            )
        ))
        

    
        fig1.update_layout(
            showlegend=True,
            plot_bgcolor='white',
            legend=dict(
                orientation='h',  # horizontal legend
                yanchor='bottom',  # anchor the legend to the bottom
                y=1.05,  # position the legend just above the bottom
                xanchor='right',  # anchor the legend to the right
                x=1  # position the legend at the far right
            )
            ),
        

        fig1.update_traces(marker=dict(size=9,
                line=dict(width=1)),
                selector=dict(mode='markers'))


        fig1.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        
        fig1.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )


        fig2 = go.Figure()
        fig2.layout.font.family = 'Arial'

        fig1.update_layout(
            title= title,
            xaxis=dict(title="Date"),
            yaxis=dict(title="Concentration (ng/l)", range=[0, 200]),
            yaxis2=dict(title="Discharge (m3/s)", overlaying="y", side="right"),
            margin=dict(l=50, r=50, t=50, b=50),  # Reduce left, right, top, and bottom margins
            
        )

        fig1.update_layout(yaxis2=dict(range=[1000, 0], side='right'))
        

        fig1.add_traces(fig2.data)
        
        
        if path is not None:
            fig1.write_image(path, width=800, height=500)

        return fig1
        
    def store_channel_csv(self, gis_id, path = None):

        predictions = self.gdf_map.copy()
        observations = self.gdf_observacions.copy()

        predictions['ng_l'] = predictions['mg_l']*1e6
        observations['valor_ng_l'] = observations['valor']*1e6

        predictions_channel = predictions[predictions['gis_id'] == gis_id][['ng_l', 'Date', 'flo_out']]
        observations_channel = observations[observations['gis_id'] == gis_id].copy()

        rmse = self.rmse
        upper_bound = predictions_channel['ng_l'] + 2*rmse
        lower_bound = predictions_channel['ng_l'] - 2*rmse
        lower_bound = lower_bound.apply(lambda x: max(0, x))
        
        
        predictions_channel['upper_bound'] = upper_bound
        predictions_channel['lower_bound'] = lower_bound
        
        predictions_channel.to_csv(path, index=False)




        

