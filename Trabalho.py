import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Função principal
def main():
    
    # Configuração da interface do Streamlit
    st.set_page_config(layout="wide")
    st.markdown("<h3 style='text-align: center;'>Trabalho de Imersão Profissional: Aplicação de Métodos de Aprendizagem de Máquina</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; font-size: 16px;'>Criado por: Leandro Quintanilha e Karina Souza</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; font-size: 14px;'>Turma FLC27184BDI</h5>", unsafe_allow_html=True)

# Página Inicial
def pagina_inicial():
    st.markdown("<div class='header'><h4>Sobre o Projeto</h4></div>", unsafe_allow_html=True)
    st.write("""
    Este projeto utiliza aprendizado de máquina para analisar dados de empréstimos com o objetivo de identificar fraudes e prever comportamentos de risco. Ao aplicar técnicas de aprendizado supervisionado, é possível detectar padrões e características que indicam a probabilidade de um empréstimo ser fraudulento, o que é essencial para prevenir perdas financeiras. Além disso, o uso de gráficos interativos facilita a exploração das variáveis envolvidas, permitindo que diferentes aspectos dos dados sejam analisados de maneira detalhada, o que aprimora a compreensão dos fatores de risco.
    
    Uma parte crucial do trabalho é a análise do histórico de empréstimos da instituição, o que fornece insights sobre o comportamento dos clientes ao longo do tempo. Esse conhecimento sobre o perfil dos solicitantes permite não apenas a identificação de empréstimos fraudulentos, mas também o mapeamento de fatores associados ao risco. A análise das variáveis e dos padrões históricos é essencial para entender como diferentes características podem influenciar a probabilidade de fraude, como comportamento de pagamento, valor dos empréstimos e dados demográficos dos clientes.
    
    A aplicação de modelos preditivos, com base nos dados históricos, gera informações valiosas que auxiliam na tomada de decisão, promovendo uma gestão mais segura e eficiente. A capacidade de antecipar fraudes e identificar riscos permite que a instituição financeira tome ações proativas, reduzindo perdas e aumentando a confiança no processo de concessão de crédito. Este trabalho destaca a importância de ferramentas analíticas e modelos preditivos no combate à fraude e na melhoria da segurança financeira.
    """)

# Carregando o arquivo uma única vez e armazenando no cache
@st.cache_data
def load_data(path='loan_data.csv'):
    try:
        dados = pd.read_csv(path)
        return dados
    except FileNotFoundError:
        st.error(f"Arquivo '{path}' não encontrado. Por favor, verifique o caminho e tente novamente.")
        return pd.DataFrame()  # Retorna um DataFrame vazio em caso de erro

# Função para exibir os dados
def display_dados(dados):
    if dados.empty:
        st.warning("Nenhum dado para exibir.")
        return
    
    st.markdown("<h2 style='color:blue;'>Tabela de Dados</h2>", unsafe_allow_html=True)
    st.write("Os dados a serem analisados:")
    st.write("Quantidade de Dados:", len(dados), "Quantidade de Colunas:", len(dados.columns), "Quantidade de Linhas:", len(dados.index))
    st.dataframe(dados)

    st.markdown("<h2 style='color:blue;'>Descrição dos Dados</h2>", unsafe_allow_html=True)
    st.write("Descrição dos dados:")
    st.write(dados.describe())
    
    st.markdown("<h2 style='color:blue;'>Tipos de Dados</h2>", unsafe_allow_html=True)
    st.write("Tipos de dados:")
    st.write(dados.dtypes)

# Função para exibir gráficos
def graficos(dados):
    if dados.empty:
        st.warning("Nenhum dado disponível para gerar gráficos.")
        return
    
    st.markdown("<h2 style='color:green;'>Gráficos dos Dados</h2>", unsafe_allow_html=True)

    # Gráfico 1: Distribuição dos valores de empréstimos
    if 'loan_amnt' in dados.columns:
        fig1 = px.bar(
            dados,
            x=dados['loan_amnt'].value_counts().index,
            y=dados['loan_amnt'].value_counts().values,
            labels={'x': 'Valor do Empréstimo', 'y': 'Montante de Empréstimos'},
            title="Distribuição dos Valores dos Empréstimos",
            color_discrete_sequence=['skyblue']
        )
        st.plotly_chart(fig1)
        st.write("No gráfico acima, podemos observar a frequência dos valores de empréstimos no conjunto de dados.")

    # Gráfico 2: Valores de empréstimos por faixa etária e grau de escolaridade
    if 'person_age' in dados.columns and 'person_education' in dados.columns:
        bins = [18, 25, 35, 50, 65, 100]
        labels = ['18-25', '26-35', '36-50', '51-65', '65+']
        dados['age_group'] = pd.cut(dados['person_age'], bins=bins, labels=labels, right=False)

        fig2 = px.bar(
            dados,
            x='age_group',
            y='loan_amnt',
            color='person_education',
            labels={'age_group': 'Faixa Etária', 'loan_amnt': 'Valor Total de Empréstimos'},
            title="Valores de Empréstimos por Idade e Grau de Escolaridade",
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        st.plotly_chart(fig2)
        st.write("No gráfico acima, podemos observar a distribuição dos valores de empréstimos por faixa etária e grau de escolaridade.")

    # Gráfico 3: Distribuição dos valores de empréstimos por faixa etária
    if 'person_age' in dados.columns:
        fig3 = px.histogram(
            dados,
            x='person_age',
            y='loan_amnt',
            labels={'person_age': 'Faixa Etária', 'loan_amnt': 'Valor Total de Empréstimos'},
            title="Distribuição dos Valores de Empréstimos por Faixa Etária",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        st.plotly_chart(fig3)
        st.write("No gráfico acima, podemos observar a distribuição dos valores de empréstimos por faixa etária.")  

# Função para treinar o modelo de regressão linear
def treinar_modelo(dados):
    if 'loan_amnt' not in dados.columns:
        st.error("Os dados não contêm a coluna 'loan_amnt'.")
        return None, None

    # Selecionar variáveis numéricas
    features = dados[['person_age']]  # Usando apenas a idade para simplificação
    target = dados['loan_amnt']

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Treinar o modelo de Regressão Linear
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Avaliar o modelo
    y_pred = modelo.predict(X_test)
    erro = mean_squared_error(y_test, y_pred, squared=False)  # Erro quadrático médio (RMSE)
    r2 = r2_score(y_test, y_pred)  # Coeficiente de determinação (R²)

    st.write(f"Erro Quadrático Médio (RMSE): {erro:.2f}")
    st.write(f"Acurácia (R²): {r2:.2%}")

    return modelo

# Função para fazer previsões
def previsoes(modelo):
    st.markdown("<h2 style='color:blue;'>Previsão de Valor de Empréstimo</h2>", unsafe_allow_html=True)
    
    # Simplificando a previsão para uma variável: idade do cliente
    idade = st.number_input("Insira a idade do cliente:", min_value=18, max_value=100, value=30)
    
    # Fazer a previsão
    entrada_df = pd.DataFrame([[idade]], columns=['person_age'])
    previsao = modelo.predict(entrada_df)[0]
    
    st.success(f"O valor previsto do empréstimo para uma pessoa de {idade} anos é: R${previsao:.2f}")

# Página de conclusão
def conclusao():
    st.markdown("<h2 style='color:orange;'>Conclusão</h2>", unsafe_allow_html=True)
    st.write("""
   Este projeto teve como objetivo analisar dados de empréstimos para identificar padrões financeiros e prever tendências relacionadas ao comportamento de concessão de crédito. Por meio de uma abordagem que combinou exploração de dados e aprendizado de máquina, foi possível extrair insights valiosos sobre os fatores que influenciam as decisões financeiras, além de compreender melhor o perfil dos solicitantes de crédito.

A análise exploratória revelou relações significativas entre variáveis, como faixas etárias, níveis de escolaridade e valores de empréstimos solicitados, servindo de base para a construção de modelos preditivos. A aplicação de algoritmos, como a regressão linear, permitiu prever montantes de empréstimos com base em características específicas, alcançando métricas que indicam um bom desempenho inicial, como o Erro Quadrático Médio (RMSE) e o coeficiente de determinação (R²).

O projeto demonstrou como a integração entre análise de dados e aprendizado de máquina pode apoiar instituições financeiras na tomada de decisões mais estratégicas e fundamentadas. Além disso, reforçou a importância de análises baseadas em dados para impulsionar inovações e melhorar a eficiência em mercados altamente dinâmicos e competitivos.
    """)

# Layout com abas
def main_layout():
    # Carregando os dados uma vez
    dados = load_data()

    # Definindo as abas
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Página Inicial", "Dados", "Gráficos", "Treinamento de Modelo", "Conclusão"])

    with tab1:
        pagina_inicial()

    with tab2:
        display_dados(dados)

    with tab3:
        graficos(dados)

    with tab4:
        modelo = treinar_modelo(dados)  # Treinamento do modelo

        if modelo:
            previsoes(modelo)

    with tab5:
        conclusao()

if __name__ == "__main__":
    main()
    main_layout()
