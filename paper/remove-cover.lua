local removed = false

function Para(el)
  if removed then return nil end
  for _, inline in ipairs(el.content) do
    if inline.t == "Image" then
      removed = true
      return {}
    end
  end
end

function Plain(el)
  if removed then return nil end
  for _, inline in ipairs(el.content) do
    if inline.t == "Image" then
      removed = true
      return {}
    end
  end
end

function RawBlock(el)
  if removed then return nil end
  -- Catch cover images embedded as raw HTML <img ...>
  if el.format == "html" or el.format == "html5" then
    if el.text:match("<img") then
      removed = true
      return {}
    end
  end
end

function Div(el)
  if removed then return nil end
  -- Some EPUBs wrap the cover in a div with a cover class
  local classes = el.classes or {}
  for _, c in ipairs(classes) do
    if c:match("cover") then
      removed = true
      return {}
    end
  end
end