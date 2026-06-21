from tkinter import Tk, Label, Button, Text, END, StringVar, messagebox, Toplevel, OptionMenu
from tkinter import ttk
from modelo import AnalisadorNoticias
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

class InterfaceAnalisador:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisador de Notícias Ambientais")
        self.root.geometry("1000x800")
        
        self.analisador = AnalisadorNoticias()
        self.criar_interface()
    
    def criar_interface(self):
        # Área de texto
        Label(self.root, text="Digite o texto da notícia:", font=('Arial', 12)).pack(pady=10)
        self.texto_noticia = Text(self.root, height=15, width=120, font=('Arial', 11))
        self.texto_noticia.pack(pady=5)
        
        # Campo de entrada para URL
        frame_inputs = ttk.Frame(self.root)
        frame_inputs.pack(pady=5)

        Label(frame_inputs, text="Digite sua URL aqui:", font=('Arial', 12)).grid(row=0, column=0, padx=5)
        self.entry_url = ttk.Entry(frame_inputs, width=100, font=('Arial', 11))
        self.entry_url.grid(row=0, column=1, padx=5)

        # Botão de análise
        Button(self.root, text="Analisar Notícia", command=self.analisar,
              font=('Arial', 12), bg='#4CAF50', fg='white').pack(pady=10)
        
        # Frame de resultados
        self.frame_resultados = ttk.LabelFrame(self.root, text="Resultados", padding=10)
        self.frame_resultados.pack(pady=10, fill='both', expand=True, padx=10)
        
        # Labels para resultados
        self.labels = {
            'protocolo': Label(self.frame_resultados, text="Protocolo: ", font=('Arial', 11)),  # Nova label do protocolo
            'assunto': Label(self.frame_resultados, text="Assunto: ", font=('Arial', 11)),
            'sentimento': Label(self.frame_resultados, text="Sentimento: ", font=('Arial', 11)),
            'percentuais': Label(self.frame_resultados, text="Percentuais: ", font=('Arial', 11))
        }
        for label in self.labels.values():
            label.pack(anchor='w')
        
        # Botões para correção
        self.frame_botoes_correcao = ttk.Frame(self.frame_resultados)
        self.frame_botoes_correcao.pack(pady=15)
        
        Button(self.frame_botoes_correcao, text="Corrigir Assunto", 
              command=self.corrigir_assunto, font=('Arial', 10)).pack(side='left', padx=5)
        
        Button(self.frame_botoes_correcao, text="Corrigir Sentimento", 
              command=self.corrigir_sentimento, font=('Arial', 10)).pack(side='left', padx=5)
        
        # Frame para os botões de ação (lado a lado)
        frame_botoes_acoes = ttk.Frame(self.root)
        frame_botoes_acoes.pack(pady=10)

        # Botão para visualizar relatório
        Button(frame_botoes_acoes, text="Visualizar Relatório", command=self.mostrar_janela_relatorio,
              font=('Arial', 12), bg='#FF9800', fg='white').pack(side='left', padx=10)

        # Botão para salvar análise
        Button(frame_botoes_acoes, text="Salvar Análise", command=self.salvar,
              font=('Arial', 12), bg='#2196F3', fg='white').pack(side='right', padx=10)
    
    def analisar(self):
        texto = self.texto_noticia.get("1.0", END).strip()
        url = self.entry_url.get().strip()
        
        if texto and url:
            messagebox.showerror("Erro", "Preencha apenas um dos campos: texto OU URL.")
            return
        elif not texto and not url:
            messagebox.showerror("Erro", "Digite o texto da notícia ou uma URL para análise.")
            return
        elif url:
            try:
                texto = self.analisador.extrair_texto_da_url(url)
                protocolo = f"{url.split('://')[0].upper()}://" if "://" in url else "Desconhecido"  # Extrai protocolo
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao extrair texto da URL: {str(e)}")
                return
        else:
            protocolo = "Texto Manual"  # Caso o usuário tenha digitado um texto, sem URL.


        self.resultado = self.analisador.analisar_noticia(texto)
        self.resultado["protocolo"] = protocolo  # Armazena protocolo no resultado
        self.atualizar_resultados()

    def atualizar_resultados(self):
        self.labels['protocolo'].config(text=f"Protocolo: {self.resultado['protocolo']}")  # Atualiza o protocolo na interface
        self.labels['assunto'].config(text=f"Assunto: {self.resultado['assunto']}")
        self.labels['sentimento'].config(text=f"Sentimento: {self.resultado['sentimento']}")
        self.labels['percentuais'].config(text=(
            f"Percentuais:\n"
            f"Positiva: {self.resultado['percentuais']['Positiva']:.2f}%\n"
            f"Negativa: {self.resultado['percentuais']['Negativa']:.2f}%\n"
            f"Neutra: {self.resultado['percentuais']['Neutra']:.2f}%"
        ))
    
    def corrigir_assunto(self):
        if not hasattr(self, 'resultado'):
            messagebox.showerror("Erro", "Analise uma notícia primeiro!")
            return
        
        janela = Toplevel(self.root)
        janela.title("Corrigir Assunto")
        
        Label(janela, text="Selecione o assunto correto:", font=('Arial', 12)).pack(pady=10)
        
        opcoes_assunto = self.analisador.label_encoder_assunto.classes_
        var_assunto = StringVar(janela, value=self.resultado['assunto'])
        
        menu_assunto = OptionMenu(janela, var_assunto, *opcoes_assunto)
        menu_assunto.config(width=20, font=('Arial', 11))
        menu_assunto.pack(pady=10)
        
        Button(janela, text="Confirmar", 
              command=lambda: [self.aplicar_correcao('assunto', var_assunto.get()), janela.destroy()],
              font=('Arial', 11), bg='#4CAF50', fg='white').pack(pady=10)
    
    def corrigir_sentimento(self):
        if not hasattr(self, 'resultado'):
            messagebox.showerror("Erro", "Analise uma notícia primeiro!")
            return
        
        janela = Toplevel(self.root)
        janela.title("Corrigir Sentimento")
        
        Label(janela, text="Selecione o sentimento correto:", font=('Arial', 12)).pack(pady=10)
        
        opcoes_sentimento = self.analisador.label_encoder_sentimento.classes_
        var_sentimento = StringVar(janela, value=self.resultado['sentimento'])
        
        menu_sentimento = OptionMenu(janela, var_sentimento, *opcoes_sentimento)
        menu_sentimento.config(width=20, font=('Arial', 11))
        menu_sentimento.pack(pady=10)
        
        Button(janela, text="Confirmar", 
              command=lambda: [self.aplicar_correcao('sentimento', var_sentimento.get()), janela.destroy()],
              font=('Arial', 11), bg='#4CAF50', fg='white').pack(pady=10)
    
    def aplicar_correcao(self, campo, valor):
        self.resultado[campo] = valor
        
        if campo == 'sentimento':
            percentuais = {'Positiva': 0, 'Negativa': 0, 'Neutra': 0}
            percentuais[valor] = 100
            self.resultado['percentuais'] = percentuais
        
        self.atualizar_resultados()
        messagebox.showinfo("Sucesso", f"{campo.capitalize()} corrigido com sucesso!")
    
    def mostrar_janela_relatorio(self):
        janela = Toplevel(self.root)
        janela.title("Relatório de Notícias")
        janela.geometry("1800x820")

        # Notebook para abas
        notebook = ttk.Notebook(janela)
        notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Aba de Resumo
        frame_resumo = ttk.Frame(notebook)
        notebook.add(frame_resumo, text="Resumo Diário")
        
        self.tree_resumo = ttk.Treeview(frame_resumo)
        self.tree_resumo.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Aba de Gráfico
        frame_grafico = ttk.Frame(notebook)
        notebook.add(frame_grafico, text="Gráfico")
        
        self.fig = Figure(figsize=(10, 4), dpi=100)
        self.ax1 = self.fig.add_subplot(121)  # Primeiro gráfico (esquerda)
        self.ax2 = self.fig.add_subplot(122)  # Segundo gráfico (direita)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_grafico)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Aba de Detalhes
        frame_detalhes = ttk.Frame(notebook)
        notebook.add(frame_detalhes, text="Detalhes")
        
        self.text_detalhes = Text(frame_detalhes, wrap='word', font=('Arial', 10))
        scroll = ttk.Scrollbar(frame_detalhes, command=self.text_detalhes.yview)
        self.text_detalhes.config(yscrollcommand=scroll.set)
        
        scroll.pack(side='right', fill='y')
        self.text_detalhes.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Carregar dados iniciais
        self.atualizar_relatorio()
    
    def atualizar_relatorio(self):
        try:
            data_inicio = "2025-05-04"
            data_fim = "2025-05-08"

            if self.analisador.df.empty:
                messagebox.showwarning("Aviso", "Nenhum dado encontrado no banco de dados")
                return

            relatorio_exibicao, df_exibicao, resumo_sentimentos = self.analisador.gerar_dados_relatorio(data_inicio, data_fim)

            if relatorio_exibicao.empty:
                messagebox.showwarning("Aviso", "Nenhum dado encontrado com os filtros atuais")
                return
            
            # Atualizar aba de resumo
            for item in self.tree_resumo.get_children():
                self.tree_resumo.delete(item)

            columns = ['Data', 'Total Notícias', '% Positiva', '% Negativa', '% Neutra']
            self.tree_resumo['columns'] = columns
            self.tree_resumo['show'] = 'headings'  # Oculta a coluna #0 (vazia à esquerda)

            for col in columns:
                self.tree_resumo.heading(col, text=col)
                self.tree_resumo.column(col, width=120, anchor='center')

            for index, row in relatorio_exibicao.iterrows():
                self.tree_resumo.insert('', 'end', values=[
                    index,
                    int(row['Total_Noticias']),
                    f"{row['Percentual_Positiva']:.1f}%",
                    f"{row['Percentual_Negativa']:.1f}%",
                    f"{row['Percentual_Neutra']:.1f}%"
                ])

            # Atualizar gráficos
            self.ax1.clear()
            self.ax2.clear()

            self.fig.subplots_adjust(wspace=0.4)

            # Gráfico 1: Evolução por dia
            relatorio_exibicao[['Percentual_Positiva', 'Percentual_Negativa', 'Percentual_Neutra']].plot(
                kind='bar', stacked=True, ax=self.ax1, color=['green', 'red', 'gray'], width=0.8)
            self.ax1.set_title('Evolução dos Sentimentos por Dia')
            self.ax1.set_ylabel('Percentual')
            self.ax1.set_xlabel('Data')
            self.ax1.legend(['Positiva', 'Negativa', 'Neutra'])
            self.ax1.grid(True, linestyle='--', alpha=0.6)

            # Gráfico 2: Sentimentos por Assunto (em barras horizontais)
            # Garantir ordem das colunas
            resumo_sentimentos = resumo_sentimentos[['Positiva', 'Negativa', 'Neutra']]
            resumo_sentimentos.plot(
                kind='barh',
                ax=self.ax2,
                color=['green', 'red', 'gray']
            )
            self.ax2.set_title('Total de Sentimentos por Assunto')
            self.ax2.set_xlabel('Quantidade')
            self.ax2.set_ylabel('Assunto')
            self.ax2.legend(['Positiva', 'Negativa', 'Neutra'])
            self.ax2.grid(True, linestyle='--', alpha=0.6)

            self.canvas.draw()

            # Atualizar detalhes
            self.text_detalhes.delete(1.0, END)
            for _, row in df_exibicao.iterrows():
                self.text_detalhes.insert(END, 
                    f"Data: {row['Data_Cadastro']}\n"
                    f"Assunto: {row['Tipo_Noticia']}\n"
                    f"Sentimento: {row['Classificação']}\n"
                    f"Positiva: {row['Percentual_Positiva']:.1f}% | "
                    f"Negativa: {row['Percentual_Negativa']:.1f}% | "
                    f"Neutra: {row['Percentual_Neutra']:.1f}%\n"
                    f"Texto: {row['Texto'][:500]}...\n"
                    f"{'-'*80}\n"
                )
        
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao atualizar relatório: {str(e)}")
    
    def salvar(self):
        if not hasattr(self, 'resultado'):
            messagebox.showerror("Erro", "Analise uma notícia primeiro!")
            return
        
        self.analisador.salvar_noticia(self.resultado)
        messagebox.showinfo("Sucesso", "Notícia salva e modelos atualizados!")
        self.texto_noticia.delete("1.0", END)

if __name__ == "__main__":
    root = Tk()
    app = InterfaceAnalisador(root)
    root.mainloop()