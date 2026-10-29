using System.Text.Json;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using HtmlAgilityPack;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;
using Microsoft.Data.Sqlite;
using Microsoft.Graph;
using Microsoft.Graph.Models;
using OllamaSharp;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.AI;

var builder = Microsoft.AspNetCore.Builder.WebApplication.CreateBuilder(args);

// Static files (HTML, CSS, JS) serve karne ke liye support
builder.Services.AddDirectoryBrowser();

// Ollama API client ko singleton register karna
builder.Services.AddSingleton(new OllamaApiClient(new Uri("http://localhost:11434")));

var app = builder.Build();

// wwwroot folder se frontend GUI files serve karna
app.UseDefaultFiles();
app.UseStaticFiles();

// Yeh direct endpoint add karein:
app.MapGet("/", () =>
{
    // Multiple possible locations check karein
    string[] possiblePaths = [
        Path.Combine(app.Environment.ContentRootPath, "wwwroot", "index.html"),
        Path.Combine(AppContext.BaseDirectory, "wwwroot", "index.html"),
        Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "index.html"),
        "wwwroot/index.html"
    ];

    foreach (var path in possiblePaths)
    {
        if (File.Exists(path))
        {
            return Results.Content(File.ReadAllText(path), "text/html");
        }
    }

    return Results.NotFound("index.html file nahi mili. Path check karein.");
});
const string DbFile = "rag.db";
string connectionString = $"Data Source={DbFile}";

// Database schema initialize karna
InitializeDatabase(connectionString);

const string embeddingModel = "nomic-embed-text";
const string llmModel = "llama3.2";

// ============================================================================
// ENDPOINT 1: Sitemap Crawl, Chunking aur Embedding Pipeline
// ============================================================================
app.MapPost("/api/crawl-sitemap", async (SitemapRequest request, OllamaApiClient ollama) =>
{
    if (string.IsNullOrWhiteSpace(request.SitemapUrl))
    {
        return Results.BadRequest(new { error = "Sitemap URL required hai." });
    }

    using var http = new HttpClient();
    http.DefaultRequestHeaders.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LocalRAGBot/1.0");

    List<string> pageUrls = new();

    // --- Part A: Sitemap XML Parse karke URLs nikaalna ---
    try
    {
        string xmlContent = await http.GetStringAsync(request.SitemapUrl);
        var xdoc = XDocument.Parse(xmlContent);

        // XML namespace se azaad rehne ke liye LocalName == "loc" check kar rahe hain
        pageUrls = xdoc.Descendants()
            .Where(e => e.Name.LocalName == "loc")
            .Select(e => e.Value.Trim())
            .Take(request.MaxPages > 0 ? request.MaxPages : 5) // Default top 5 pages limit
            .ToList();
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new { error = $"Sitemap parse karne mein masla aya: {ex.Message}" });
    }

    if (pageUrls.Count == 0)
    {
        return Results.BadRequest(new { error = "Sitemap mein koi URL nahi mila." });
    }

    int totalChunksCreated = 0;
    var processedPages = new List<PageProcessResult>();

    ollama.SelectedModel = embeddingModel;

    // --- Part B: Har Page ko Crawl, Clean aur Chunk karna ---
    foreach (var url in pageUrls)
    {
        try
        {
            string html = await http.GetStringAsync(url);
            string cleanText = ExtractCleanTextFromHtml(html);

            if (string.IsNullOrWhiteSpace(cleanText) || cleanText.Length < 100)
            {
                continue; // Khali ya bohot chotay pages ko skip karein
            }

            // Text ko 800 characters ke chunks mein split karna with 150 characters overlap
            var chunks = CreateChunksWithOverlap(cleanText, chunkSize: 800, overlap: 150);

            // Har chunk ka embedding generate karke SQLite mein save karna
            for (int i = 0; i < chunks.Count; i++)
            {
                var chunkText = chunks[i];
                var embedResponse = await ollama.EmbedAsync(chunkText);
                if (embedResponse?.Embeddings == null || embedResponse.Embeddings.Count == 0)
                {
                    continue;
                }
                float[] embedding = embedResponse.Embeddings[0];

                SaveChunkToDb(connectionString, url, i + 1, chunkText, embedding);
                totalChunksCreated++;
            }

            processedPages.Add(new PageProcessResult(url, chunks.Count, cleanText.Length));
        }
        catch
        {
            // Kisi page ke fail hone par aglay page par move karein
        }
    }

    return Results.Ok(new
    {
        message = "Sitemap crawling aur chunking complete ho gayi!",
        totalPagesCrawled = processedPages.Count,
        totalChunks = totalChunksCreated,
        details = processedPages
    });
});

// ============================================================================
// ENDPOINT 2: RAG Chat (Vector Similarity Search + Llama 3.2 Response)
// ============================================================================
app.MapPost("/api/chat", async (ChatRequest request, OllamaApiClient ollama) =>
{
    if (string.IsNullOrWhiteSpace(request.Query))
    {
        return Results.BadRequest(new { error = "Query required hai." });
    }

    var allChunks = LoadAllChunksFromDb(connectionString);
    if (allChunks.Count == 0)
    {
        return Results.BadRequest(new { error = "Database khali hai. Pehle sitemap crawl karein." });
    }

    // 1. User Query ka embedding banana
    ollama.SelectedModel = embeddingModel;
    var queryResponse = await ollama.EmbedAsync(request.Query);
    if (queryResponse?.Embeddings == null || queryResponse.Embeddings.Count == 0)
    {
        return Results.BadRequest(new { error = "Embedding generate nahi ho saki." });
    }
    float[] queryEmbedding = queryResponse.Embeddings[0];

    // 2. Cosine similarity se top 3 relevant chunks dhoondna
    var topResults = allChunks
        .Select(chunk => new
        {
            Chunk = chunk,
            Score = CosineSimilarity(queryEmbedding, chunk.Embedding)
        })
        .OrderByDescending(r => r.Score)
        .Take(3)
        .ToList();

    // 3. Prompt mein Context aur Sources inject karna
    string contextText = string.Join("\n\n", topResults.Select(r =>
        $"[Source: {r.Chunk.Url} | Chunk #{r.Chunk.ChunkIndex}]\n{r.Chunk.Text}"));

    string systemPrompt = $"""
        You are an intelligent knowledge assistant for the website.
        Answer the user's question accurately using ONLY the context provided below.
        Always cite the source URL from where the information came.
        If the answer is not in the context, say "I cannot find this information on the indexed pages."

        Context:
        {contextText}
        """;

    string answer = "";
    ollama.SelectedModel = llmModel;
    var generateRequest = new OllamaSharp.Models.GenerateRequest
    {
        Prompt = request.Query,
        System = systemPrompt
    };
    await foreach (var token in ollama.GenerateAsync(generateRequest))
    {
        if (token?.Response != null)
            answer += token.Response;
    }

    return Results.Ok(new
    {
        query = request.Query,
        answer = answer.Trim(),
        sources = topResults.Select(r => new { r.Chunk.Url, r.Chunk.ChunkIndex, r.Score, r.Chunk.Text })
    });
});

// ============================================================================
// ENDPOINT 3: System Stats (Total pages & chunks view)
// ============================================================================
app.MapGet("/api/stats", () =>
{
    var chunks = LoadAllChunksFromDb(connectionString);
    var uniqueUrls = chunks.Select(c => c.Url).Distinct().Count();

    return Results.Ok(new
    {
        totalChunks = chunks.Count,
        totalPages = uniqueUrls,
        recentChunks = chunks.TakeLast(5).Select(c => new { c.Url, c.ChunkIndex, Preview = c.Text.Substring(0, Math.Min(120, c.Text.Length)) + "..." })
    });
});

app.Run();

// ============================================================================
// CORE HELPER FUNCTIONS (Chunking, Cleaning & DB)
// ============================================================================

/// <summary>
/// Sliding-Window Chunking with Overlap.
/// Overlap se sentences aur context boundary break nahi hoti.
/// </summary>
static List<string> CreateChunksWithOverlap(string text, int chunkSize, int overlap)
{
    var chunks = new List<string>();
    if (string.IsNullOrEmpty(text)) return chunks;

    int startIndex = 0;
    while (startIndex < text.Length)
    {
        int length = Math.Min(chunkSize, text.Length - startIndex);
        string chunk = text.Substring(startIndex, length).Trim();

        if (!string.IsNullOrWhiteSpace(chunk))
        {
            chunks.Add(chunk);
        }

        // Aglay chunk ke liye window slide karein
        startIndex += (chunkSize - overlap);

        if (startIndex + overlap >= text.Length)
        {
            // Aakhri bacha hua part agar chota ho to cover karein
            if (startIndex < text.Length)
            {
                string lastChunk = text.Substring(startIndex).Trim();
                if (!string.IsNullOrWhiteSpace(lastChunk)) chunks.Add(lastChunk);
            }
            break;
        }
    }

    return chunks;
}

/// <summary>
/// HTML se navigation, footer, scripts hata kar saaf content nikaalna
/// </summary>
static string ExtractCleanTextFromHtml(string html)
{
    var doc = new HtmlDocument();
    doc.LoadHtml(html);

    // Unnecessary HTML tags ko remove karna
    var nodesToRemove = doc.DocumentNode.SelectNodes("//script|//style|//nav|//footer|//header|//aside|//noscript");
    if (nodesToRemove != null)
    {
        foreach (var node in nodesToRemove) node.Remove();
    }

    string innerText = doc.DocumentNode.InnerText;

    // HTML entities decode karna (jaise &nbsp; ya &amp;)
    string decoded = System.Net.WebUtility.HtmlDecode(innerText);

    // Multiple linebreaks aur tabs ko single clean space mein convert karna
    return Regex.Replace(decoded, @"\s+", " ").Trim();
}

static void InitializeDatabase(string connStr)
{
    using var connection = new SqliteConnection(connStr);
    connection.Open();

    string sql = """
        CREATE TABLE IF NOT EXISTS WebChunks (
            Id INTEGER PRIMARY KEY AUTOINCREMENT,
            Url TEXT NOT NULL,
            ChunkIndex INTEGER NOT NULL,
            Text TEXT NOT NULL,
            EmbeddingJson TEXT NOT NULL
        );
        """;
    using var cmd = new SqliteCommand(sql, connection);
    cmd.ExecuteNonQuery();
}

static void SaveChunkToDb(string connStr, string url, int chunkIndex, string text, float[] embedding)
{
    using var connection = new SqliteConnection(connStr);
    connection.Open();

    string sql = "INSERT INTO WebChunks (Url, ChunkIndex, Text, EmbeddingJson) VALUES (@url, @index, @text, @embedding);";
    using var cmd = new SqliteCommand(sql, connection);
    cmd.Parameters.AddWithValue("@url", url);
    cmd.Parameters.AddWithValue("@index", chunkIndex);
    cmd.Parameters.AddWithValue("@text", text);
    cmd.Parameters.AddWithValue("@embedding", JsonSerializer.Serialize(embedding));
    cmd.ExecuteNonQuery();
}

static List<WebDocumentChunk> LoadAllChunksFromDb(string connStr)
{
    var list = new List<WebDocumentChunk>();
    using var connection = new SqliteConnection(connStr);
    connection.Open();

    string sql = "SELECT Url, ChunkIndex, Text, EmbeddingJson FROM WebChunks;";
    using var cmd = new SqliteCommand(sql, connection);
    using var reader = cmd.ExecuteReader();

    while (reader.Read())
    {
        string url = reader.GetString(0);
        int index = reader.GetInt32(1);
        string text = reader.GetString(2);
        string json = reader.GetString(3);
        var embedding = JsonSerializer.Deserialize<float[]>(json);

        if (embedding != null)
        {
            list.Add(new WebDocumentChunk
            {
                Url = url,
                ChunkIndex = index,
                Text = text,
                Embedding = embedding
            });
        }
    }
    return list;
}

static float CosineSimilarity(float[] vectorA, float[] vectorB)
{
    if (vectorA == null || vectorB == null || vectorA.Length != vectorB.Length || vectorA.Length == 0)
    {
        return 0f;
    }

    float dotProduct = 0f, normA = 0f, normB = 0f;
    for (int i = 0; i < vectorA.Length; i++)
    {
        dotProduct += vectorA[i] * vectorB[i];
        normA += vectorA[i] * vectorA[i];
        normB += vectorB[i] * vectorB[i];
    }
    float denominator = MathF.Sqrt(normA) * MathF.Sqrt(normB);
    return denominator == 0 ? 0 : dotProduct / denominator;
}

// Request & Data DTOs
record SitemapRequest(string SitemapUrl, int MaxPages = 5);
record ChatRequest(string Query);
record PageProcessResult(string Url, int ChunksCount, int TextLength);

class WebDocumentChunk
{
    public required string Url { get; set; }
    public required int ChunkIndex { get; set; }
    public required string Text { get; set; }
    public required float[] Embedding { get; set; }
}
