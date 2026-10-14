# Budget alert (optional): e-mails the billing admins at 50 %, 90 % and 100 % of a small
# monthly amount, so a misconfiguration (min instances, a runaway job) is noticed in days.
data "google_project" "this" {}

resource "google_billing_budget" "monthly" {
  count = var.billing_account == null ? 0 : 1

  billing_account = var.billing_account
  display_name    = "brent-forecast monthly"

  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }

  amount {
    specified_amount {
      currency_code = var.budget.currency
      units         = tostring(var.budget.amount)
    }
  }

  dynamic "threshold_rules" {
    for_each = [0.5, 0.9, 1.0]
    content {
      threshold_percent = threshold_rules.value
    }
  }
}
