# ============================================================================
#  LendGuard - AI-assisted credit risk decisioning
#  Loads the trained XGBoost model, scores a borrower, finds the factors that
#  drove this decision (SHAP), and asks Claude to write a plain-English notice.
#
#  FILES NEEDED IN THIS FOLDER:  app.R, lendguard_xgb.model, lendguard_meta.rds
#  RUN:  setwd("path/to/this/folder"); shiny::runApp(launch.browser = TRUE)
# ============================================================================

library(shiny)
library(xgboost)
library(caret)
library(ellmer)

meta    <- readRDS("lendguard_meta.rds")
booster <- xgb.load("lendguard_xgb.model")

map_to_orig <- function(colname) {
  hit <- meta$features[startsWith(colname, meta$features)]
  if (length(hit) == 0) colname else hit[which.max(nchar(hit))]
}

readable_label <- c(
  person_home_ownership      = "home ownership status",
  loan_intent                = "loan purpose",
  loan_amnt                  = "loan amount",
  loan_int_rate              = "interest rate",
  loan_percent_income        = "loan-to-income ratio",
  cb_person_cred_hist_length = "credit history length"
)

ui <- fluidPage(
  tags$head(tags$style(HTML("
    body { font-family: system-ui, sans-serif; }
    .brand { font-size: 22px; font-weight: 600; margin-bottom: 2px; }
    .sub { color: #666; margin-bottom: 18px; }
    .scorebox { padding: 14px 16px; border-radius: 10px; margin-bottom: 14px; }
    .approve { background: #E1F5EE; color: #085041; }
    .decline { background: #FCEBEB; color: #791F1F; }
    .notice { background: #fafafa; border: 1px solid #eee;
              border-radius: 10px; padding: 16px; line-height: 1.55; }
  "))),
  div(class = "brand", "LendGuard"),
  div(class = "sub", "Credit risk assessment with an automated, plain-English decision notice."),
  sidebarLayout(
    sidebarPanel(
      selectInput("home", "Home ownership", choices = meta$levels$person_home_ownership),
      selectInput("intent", "Loan purpose", choices = meta$levels$loan_intent),
      numericInput("amnt", "Loan amount ($)", value = 12000, min = 500, step = 500),
      numericInput("rate", "Interest rate (%)", value = 12.5, min = 1, step = 0.1),
      numericInput("income", "Annual income ($)", value = 55000, min = 1, step = 1000),
      numericInput("hist", "Credit history length (years)", value = 5, min = 0, step = 1),
      actionButton("go", "Assess applicant", class = "btn-primary")
    ),
    mainPanel(
      uiOutput("score"),
      h4("Decision notice"),
      div(class = "notice", uiOutput("notice"))
    )
  )
)

server <- function(input, output, session) {
  
  result <- eventReactive(input$go, {
    validate(need(input$income > 0, "Annual income must be greater than 0."))
    
    pct <- input$amnt / input$income
    
    row <- data.frame(
      person_home_ownership = factor(input$home, levels = meta$levels$person_home_ownership),
      loan_intent           = factor(input$intent, levels = meta$levels$loan_intent),
      loan_amnt             = as.numeric(input$amnt),
      loan_int_rate         = as.numeric(input$rate),
      loan_percent_income   = as.numeric(pct),
      cb_person_cred_hist_length = as.numeric(input$hist)
    )[, meta$features]
    
    x    <- predict(meta$dummies, row)
    dmat <- xgb.DMatrix(as.matrix(x))
    prob <- as.numeric(predict(booster, dmat))
    
    drivers <- tryCatch({
      contrib <- predict(booster, dmat, predcontrib = TRUE)[1, ]
      contrib <- contrib[names(contrib) != "BIAS"]
      agg <- tapply(contrib, vapply(names(contrib), map_to_orig, character(1)), sum)
      agg <- agg[names(agg) %in% meta$features]   # drop base value / any stray column
      head(names(sort(agg, decreasing = TRUE)), 4)
    }, error = function(e) meta$features[1:4])
    
    list(prob = prob, pct = pct, drivers = drivers)
  })
  
  output$score <- renderUI({
    r <- result()
    decline <- r$prob >= meta$threshold
    cls <- if (decline) "scorebox decline" else "scorebox approve"
    verdict <- if (decline) "Decline / elevated risk" else "Approve / acceptable risk"
    div(class = cls,
        strong(verdict), br(),
        sprintf("Estimated default probability: %.0f%%  (decision threshold %.0f%%)",
                100 * r$prob, 100 * meta$threshold))
  })
  
  output$notice <- renderUI({
    r <- result()
    decline <- r$prob >= meta$threshold
    
    profile <- sprintf(
      "- Home ownership: %s\n- Loan purpose: %s\n- Loan amount: $%s\n- Interest rate: %s%%\n- Loan-to-income ratio: %.0f%%\n- Credit history length: %s years",
      input$home, input$intent, format(input$amnt, big.mark = ","),
      input$rate, 100 * r$pct, input$hist
    )
    labs <- readable_label[r$drivers]
    driver_text <- paste(labs[!is.na(labs)], collapse = ", ")
    
    user_prompt <- sprintf(
      "Write a credit decision notice for this applicant.\n\nDECISION: %s\nModel default probability: %.0f%%\n\nApplicant profile:\n%s\n\nThe model's strongest risk-increasing factors for this applicant, in order: %s.\n\nWrite 4-6 sentences. If declined, give up to 4 specific principal reasons that reference the applicant's actual figures above. If approved, briefly note the main supporting factors. Plain language, no jargon.",
      if (decline) "DECLINE" else "APPROVE", 100 * r$prob, profile, driver_text
    )
    
    sys_prompt <- paste(
      "You are a lending compliance assistant. You draft clear, specific,",
      "plain-English credit decision notices consistent with US ECOA/Regulation B",
      "adverse action requirements. Cite the applicant's actual figures, use at",
      "most four principal reasons, never invent data, and avoid jargon.",
      "Format with light Markdown: a short bold decision line and well-spaced",
      "paragraphs. This is a decision-support draft, not legal advice."
    )
    
    if (nchar(Sys.getenv("ANTHROPIC_API_KEY")) == 0) {
      return(markdown(sprintf(
        "[Set ANTHROPIC_API_KEY to generate the AI notice.]\n\nDecision: %s. Estimated default probability %.0f%%. Main factors for this applicant: %s.",
        if (decline) "Declined" else "Approved", 100 * r$prob, driver_text)))
    }
    
    withProgress(message = "Generating decision notice...", value = 0.5, {
      chat <- chat_anthropic(system_prompt = sys_prompt, model = "claude-sonnet-4-6")
      markdown(chat$chat(user_prompt, echo = FALSE))
    })
  })
}

shinyApp(ui, server)