import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
from prophet.serialize import model_from_json

# Força o encoding UTF-8 no terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# --- CONFIGURAÇÃO DOS CAMINHOS ---
# Certifique-se de que esta pasta existe e contém os JSONs e o CSV
PASTA_DADOS = "api_data" # Ou 'modelos_prophet_json' se você não mudou o nome
ARQUIVO_CSV = os.path.join(PASTA_DADOS, "df_total_consolidado.csv")

# ──────────────────────────────────────────────
# 1. Carregamento dos modelos e dados
# ──────────────────────────────────────────────

def _carregar_todos_prophets():
    """Lê os arquivos JSON e reconstrói os objetos Prophet."""
    modelos = {}
    if not os.path.isdir(PASTA_DADOS):
        print(f"⚠️ Pasta '{PASTA_DADOS}' não encontrada. Crie a pasta e coloque os arquivos lá!")
        return modelos

    for arquivo in os.listdir(PASTA_DADOS):
        if arquivo.startswith("modelo_prophet_") and arquivo.endswith(".json"):
            chave = arquivo.replace("modelo_prophet_", "").replace(".json", "").upper()
            caminho = os.path.join(PASTA_DADOS, arquivo)
            
            try:
                with open(caminho, "r", encoding="utf-8") as f:
                    # Tenta ler o JSON. Se foi salvo com json.dump no Notebook, será uma string.
                    conteudo = json.load(f)
                    
                    if isinstance(conteudo, str):
                        # Caso da dupla serialização
                        modelos[chave] = model_from_json(conteudo)
                    else:
                        # Caso tenha sido salvo corretamente como arquivo texto
                        f.seek(0)
                        modelos[chave] = model_from_json(f.read())
                        
                print(f"   ✅ Prophet carregado: {chave}")
            except Exception as e:
                print(f"   ❌ Erro ao carregar {arquivo}: {e}")
            
    return modelos

def _carregar_dados_historicos():
    """Lê o CSV exato gerado pelo Notebook (colunas: ds, uf, total_cotas_ativas)."""
    if not os.path.exists(ARQUIVO_CSV):
        print(f"⚠️ Arquivo '{ARQUIVO_CSV}' não encontrado.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(ARQUIVO_CSV)
        
        # Mapeamento para o padrão que a API precisa
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.rename(columns={
            'uf': 'estado',
            'total_cotas_ativas': 'y'
        })
        
        df['estado'] = df['estado'].astype(str).str.upper()
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        return df[['estado', 'ds', 'y']].sort_values(['estado', 'ds'])

    except Exception as e:
        print(f"⚠️ Erro ao processar o CSV histórico: {e}")
        return pd.DataFrame()


print("\n🚀 Carregando modelos em memória...")
PROPHETS     = _carregar_todos_prophets()
DF_HISTORICO = _carregar_dados_historicos()
print(f"\n✅ API pronta — {len(PROPHETS)} Modelos de Estado carregados\n")

# ──────────────────────────────────────────────
# 2. Helpers compartilhados
# ──────────────────────────────────────────────

def _float_or_none(val):
    if val is None or pd.isna(val):
        return None
    try:
        v = float(val)
        return None if (np.isnan(v) or np.isinf(v)) else v
    except (TypeError, ValueError):
        return None

def _montar_resposta_prophet(df_historico, modelo, meses):
    """Gera uma linha do tempo contínua para o Frontend não quebrar."""
    data_min = df_historico['ds'].min()
    data_max = df_historico['ds'].max()
    
    # CRÍTICO: Criar um range de datas contínuo, mês a mês. 
    # Isso permite que a lógica de "% 3" do seu Javascript funcione perfeitamente.
    datas_completas = pd.date_range(
        start=data_min, 
        end=data_max + pd.DateOffset(months=meses), 
        freq='MS' # Month Start
    )
    df_futuro_continuo = pd.DataFrame({'ds': datas_completas})
    
    # Faz a previsão para TODOS os meses (passado e futuro)
    previsao = modelo.predict(df_futuro_continuo)
    
    # Junta com o histórico real (que só tem dados a cada 3 meses)
    df_junto = pd.merge(previsao, df_historico, on="ds", how="left")
    
    labels    = df_junto["ds"].dt.strftime("%m/%Y").tolist()
    historico = [_float_or_none(x) for x in df_junto["y"].tolist()]
    prev_base = [_float_or_none(x) for x in df_junto["yhat"].tolist()]
    prev_min  = [_float_or_none(x) for x in df_junto["yhat_lower"].tolist()]
    prev_max  = [_float_or_none(x) for x in df_junto["yhat_upper"].tolist()]

    # Apaga as linhas de previsão onde já existe dado histórico real
    ultimo_idx_real = -1
    for i in range(len(historico)):
        if historico[i] is not None:
            prev_base[i] = prev_min[i] = prev_max[i] = None
            ultimo_idx_real = i

    # Conecta o gráfico no último ponto real para a linha não ficar "quebrada"
    if ultimo_idx_real >= 0:
        prev_base[ultimo_idx_real] = historico[ultimo_idx_real]
        prev_min[ultimo_idx_real]  = historico[ultimo_idx_real]
        prev_max[ultimo_idx_real]  = historico[ultimo_idx_real]

    return {
        "labels":    labels,
        "historico": historico,
        "prevBase":  prev_base,
        "prevMin":   prev_min,
        "prevMax":   prev_max,
        "modelo":    "Prophet Puro"
    }

# ──────────────────────────────────────────────
# 3. Rotas
# ──────────────────────────────────────────────

@app.route("/api/prever/<estado>/<int:meses>")
def prever_prophet(estado, meses):
    estado = estado.upper()

    if estado not in PROPHETS:
        return jsonify({"erro": f"Estado '{estado}' não possui modelo Prophet."}), 404
    if DF_HISTORICO.empty:
        return jsonify({"erro": "Base de dados histórica não carregada."}), 500

    modelo = PROPHETS[estado]
    df_alvo = DF_HISTORICO[DF_HISTORICO["estado"] == estado]

    print(f"📍 Calculando: Prophet → {estado} | {meses} meses Fwd")
    
    resposta = _montar_resposta_prophet(df_alvo, modelo, meses)
    return jsonify(resposta)
@app.route("/api/ranking/<int:meses>")
def ranking_estados(meses):
    """Calcula a previsão de todos os estados e retorna o ranking dos que mais vão crescer."""
    if DF_HISTORICO.empty:
        return jsonify({"erro": "Base de dados histórica não carregada."}), 500

    ranking = []
    
    # Passa por todos os 27 modelos treinados na memória
    for uf, modelo in PROPHETS.items():
        df_alvo = DF_HISTORICO[DF_HISTORICO["estado"] == uf]
        if df_alvo.empty: 
            continue
        
        # Pega o último valor real válido histórico do Estado
        ultimo_real = float(df_alvo['y'].dropna().iloc[-1])
        
        # Projeta o futuro
        futuro = modelo.make_future_dataframe(periods=meses, freq="MS")
        previsao = modelo.predict(futuro)
        
        # Pega o valor projetado lá no final do período solicitado
        valor_projetado = float(previsao['yhat'].iloc[-1])
        
        # Calcula a porcentagem de crescimento
        if ultimo_real > 0:
            crescimento_pct = ((valor_projetado - ultimo_real) / ultimo_real) * 100
        else:
            crescimento_pct = 0.0
            
        ranking.append({
            "estado": uf,
            "crescimento_pct": round(crescimento_pct, 2),
            "projetado": round(valor_projetado),
            "atual": round(ultimo_real)
        })
        
    # Ordena a lista do maior crescimento (%) para o menor
    ranking.sort(key=lambda x: x["crescimento_pct"], reverse=True)
    
    return jsonify(ranking)

@app.route("/api/modelos/status")
def status_modelos():
    """Rota para você testar se a API subiu corretamente pelo navegador"""
    return jsonify({
        "status": "online",
        "prophets_carregados": sorted(PROPHETS.keys()),
        "linhas_historico": len(DF_HISTORICO)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)