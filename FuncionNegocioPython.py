import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.base import BaseEstimator
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

# Función principal de análisis de marketing
def analisis_marketing(
    modelo: BaseEstimator,
    datos: pd.DataFrame,
    priorizar: str = 'FN',
    coste_llamada: float = 0.0,
    coste_email: float = 2.0,
    coste_vip: float = 200.0,
    coste_oferta: float = 150.0,
    entrada: float = 0.2,
    guardar_excel: bool = False,
    ruta_salida: str = '.'
) -> pd.DataFrame:
    # Validación
    if priorizar not in ['FP', 'FN']:
        raise ValueError("Solo 'FP' o 'FN'")

    # Scoring
    datos = datos.copy()
    if hasattr(modelo, 'predict_proba'):
        datos['yhat'] = modelo.predict_proba(datos)[:, 1]
    else:
        datos['yhat'] = modelo.predict(datos)

    # Binarización
    threshold = 0.35 if priorizar=='FN' else 0.65
    datos['yhat_binary'] = (datos['yhat'] > threshold).astype(int)
    datos['rank'] = datos['yhat'].rank()

    # Lógica FN
    if priorizar == 'FN':
        df = datos.copy()
        df['rank_desc'] = df['yhat'].rank(ascending=False)

        # Grupos
        qs = df.loc[df['yhat_binary']==1, 'rank_desc'].quantile([0.33,0.66]).values
        def asignar(row):
            if row['yhat_binary']==0:
                return 'GRUPO 4' if row['balance']>70000 else 'GRUPO 5'
            if row['rank_desc'] <= qs[0]: return 'GRUPO 1'
            if row['rank_desc'] <= qs[1]: return 'GRUPO 2'
            return 'GRUPO 3'
        df['Grupo'] = df.apply(asignar, axis=1)

        # Acción y métricas
        acciones = {
            'GRUPO 1':'Llamada',
            'GRUPO 2':'Llamada + oferta pequeña',
            'GRUPO 3':'Llamada + oferta VIP',
            'GRUPO 4':'Llamada VIP hoy',
            'GRUPO 5':'Email'
        }
        df['Accion'] = df['Grupo'].map(acciones).fillna('No contactar')
        def coste(a):
            if a=='Llamada': return coste_llamada
            if 'oferta pequeña' in a: return coste_llamada+coste_oferta
            if 'VIP' in a: return coste_llamada+coste_vip
            if a=='Email': return coste_email
            return 0
        df['Coste'] = df['Accion'].apply(coste)
        df['Beneficio_Esperado'] = df['balance'] * entrada
        df['Beneficio_Neto'] = df['Beneficio_Esperado'] - df['Coste']

        df = df.sort_values('rank_desc').drop(['yhat_binary','rank','rank_desc'], axis=1)

    # Lógica FP
    else:
        df = datos.copy()
        si = df[df['yhat_binary']==1].copy()
        no = df[df['yhat_binary']==0].copy()

        # Mitades
        m = len(si)//2
        si['Accion'] = ['Llamada']*m + ['Llamada + Oferta']*(len(si)-m)
        no['Accion'] = ['WhatsApp sin coste']*(len(no)//2) + ['Correo sin coste']*(len(no)-len(no)//2)

        df = pd.concat([si,no], ignore_index=True)
        def coste_fp(a):
            if a=='Llamada': return coste_llamada
            if 'Oferta' in a: return coste_llamada+coste_oferta
            return 0
        df['Coste'] = df['Accion'].apply(coste_fp)
        df['Beneficio_Esperado'] = df['balance'] * entrada
        df['Beneficio_Neto'] = df['Beneficio_Esperado'] - df['Coste']

        df = df.sort_values('rank').drop(['yhat_binary','rank'], axis=1)

    # Guardar Excel
    if guardar_excel:
        fn = f"resultado_{priorizar}_{pd.Timestamp.today().date()}.xlsx"
        path = os.path.join(ruta_salida, fn)
        df.to_excel(path, index=False)
        print(f"Guardado: {path}")

    return df

###################################### Funciones avanzadas ######################################


#sirve para ver en q punto de clientes contactados se ganan más beneficios y si llega un momento en el que no vale la pena seguir llamando
#ordenada de mayor a menor yhat y calcula el beneficio neto acumulado / núm de clientes
def calcular_roi_marginal(df, entrada, coste_col='Coste'):
    df_local = df.copy().sort_values('yhat', ascending=False)
    df_local['Beneficio_Neto'] = df_local['balance']*entrada - df_local[coste_col]
    acum = df_local['Beneficio_Neto'].cumsum()
    roi = acum / (np.arange(len(df_local))+1)
    plt.figure()
    plt.plot(range(1,len(roi)+1), roi)
    plt.title('ROI marginal vs # clientes')
    plt.xlabel('Clientes contactados')
    plt.ylabel('ROI medio (€)')
    plt.tight_layout()
    plt.show()


#sirve para separar por meses y ver el beneficio de los clientes contactados en cada mes
def analisis_cohortes(df, month_col='month'):
    if month_col not in df.columns:
        print(f"No existe columna '{month_col}'")
        return

    # Convertir el mes (ej. 'jan', 'feb', ...) a número
    meses_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df = df.copy()
    df['mes_num'] = df[month_col].map(meses_map)

    resumen = df.groupby('mes_num')['Beneficio_Neto'].sum().sort_index()

    print("Beneficio neto por mes:")
    for mes, beneficio in resumen.items():
        print(f"Mes {mes:02d}: {beneficio:.2f}")

    # Opcional: gráfica
    plt.figure()
    plt.bar(resumen.index, resumen.values, tick_label=[f"{m:02d}" for m in resumen.index])
    plt.title("Beneficio Neto por Mes")
    plt.xlabel("Mes")
    plt.ylabel("Beneficio (€)")
    plt.tight_layout()
    plt.show()


#sirve para ver qué efecto tiene cada tipo de oferta en clientes
#divide a los clientes en dos grupos aleatorios y les aplica la oferta A (10% p.ej) a un grupo y al otro la B y compara
def setup_ab_test(df, seed=None):
    """
    Asigna aleatoriamente a A/B y muestra resumen de métricas.
    """
    if seed is not None:
        np.random.seed(seed)
    df['AB_Group'] = np.random.choice(['A','B'], size=len(df))
    # Conteo de cada grupo
    counts = df['AB_Group'].value_counts().to_dict()
    print("Recuento A/B:", counts)
    # Métricas por grupo
    if 'yhat' in df.columns and 'Beneficio_Neto' in df.columns:
        resumen = df.groupby('AB_Group').agg(
            n=('yhat','count'),
            yhat_mean=('yhat','mean'),
            balance_mean=('balance','mean'),
            beneficio_mean=('Beneficio_Neto','mean')
        )
        print("Resumen por grupo A/B:")
        print(resumen)
    # Primeros 5 clientes con su grupo
    print("Primeros 5 asignados:")
    print(df[['balance','yhat','AB_Group']].head())


#sirve para detectar si hay drifts (cambios súbitos en yhat) q podrían inidicar q el modelo no pilla bien la tendencia 
def monitor_drift(yhat, window=100, umbral=0.05):
    if len(yhat) < window:
        print("Datos insuficientes para drift.")
        return
    mov = yhat.rolling(window).mean()
    dif = mov.diff().abs()
    drift = dif[dif>umbral]
    if not drift.empty:
        idx = drift.idxmax()
        print(f"Drift en idx {idx}, cambio={drift.max():.3f}")
    else:
        print("No se detectó drift.")

# Ejemplo de uso en Spyder (ejecutar todo el archivo)
if __name__ == '__main__':
    # 1) Cargar y preprocesar datos
    datos = pd.read_csv('bank.csv')
    bin_map = {'yes':1,'no':0}
    for c in ['default','housing','loan','deposit']:
        datos[c] = datos[c].map(bin_map)
    for c in ['job','marital','education','contact','month','poutcome']:
        datos[c] = datos[c].astype('category')

    # 2) Entrenamiento GLM y wrapper
    train_df, test_df = train_test_split(datos, test_size=0.2,
                                         stratify=datos['deposit'], random_state=42)
    formula = (
        'deposit ~ age + C(job) + C(marital)*C(education) + default + '
        'balance + housing*loan + C(contact) + C(poutcome) + day*month + '
        'campaign*previous*pdays'
    )
    glm_model = smf.glm(formula=formula, data=train_df,
                        family=sm.families.Binomial()).fit(disp=False)
    class GLMWrapper:
        def __init__(self, model): self.model = model
        def predict(self, X): return self.model.predict(X)
        def predict_proba(self, X):
            p = self.model.predict(X)
            return np.vstack((1-p,p)).T
    modelo_wrap = GLMWrapper(glm_model)

    # 3) Ejecutar análisis de marketing
    df_result = analisis_marketing(modelo_wrap, test_df,
                                   priorizar='FP', coste_llamada=1.0,
                                   coste_email=2.0, coste_vip=50.0,
                                   coste_oferta=20.0, entrada=0.2,
                                   guardar_excel=False)

    # 4) Funciones avanzadas
    calcular_roi_marginal(df_result, entrada=0.2)
    analisis_cohortes(df_result, month_col='month')
    setup_ab_test(df_result, seed=42)
    df_result['Descuento'] = df_result['AB_Group'].map({'A':0.10,'B':0.15})
    df_result['Precio_Con_Oferta'] = df_result['balance'] * (1 - df_result['Descuento'])
    df_result['Beneficio_Esperado_Oferta'] = df_result['Precio_Con_Oferta'] * df_result['yhat']
    monitor_drift(df_result['yhat'], window=200, umbral=0.03)

    print("Pipeline completado con éxito.")


